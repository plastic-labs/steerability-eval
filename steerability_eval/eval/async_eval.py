import asyncio
from typing import Dict, Optional, Any
import json
import os
from datetime import datetime
from pathlib import Path

from steerability_eval.eval.base import (
    BaseEval, BaseSteerableSystem, BaseSteeredSystem, Persona, MAX_CONCURRENT_TESTS, MAX_OBSERVATIONS
)
from steerability_eval.dataset.base import BaseDataset, Observation, PersonaId
from steerability_eval.steerable.base import SteeredSystemState
from steerability_eval.eval.config import EvalConfig


class AsyncSteerabilityEval(BaseEval):
    def __init__(self, 
                 tested_system: BaseSteerableSystem, 
                 dataset: BaseDataset,
                 experiment_name: Optional[str] = None,
                 n_steer_observations_per_persona: int = 5,
                 max_observations: int = MAX_OBSERVATIONS,
                 verbose: bool = False,
                 output_base_dir: str = 'output/experiments',
                 config: EvalConfig = EvalConfig()):
        super().__init__(
            tested_system=tested_system,
            dataset=dataset,
            experiment_name=experiment_name,
            max_observations=max_observations,
            n_steer_observations_per_persona=n_steer_observations_per_persona,
            verbose=verbose,
            output_base_dir=output_base_dir,
            config=config
        )

    @classmethod
    async def create(
        cls,
        tested_system: BaseSteerableSystem,
        dataset: BaseDataset,
        config: EvalConfig,
    ) -> 'AsyncSteerabilityEval':
        instance = cls(
            tested_system=tested_system,
            dataset=dataset,
            experiment_name=config.experiment_name,
            n_steer_observations_per_persona=config.n_steer_observations_per_persona,
            max_observations=config.max_observations,
            verbose=config.verbose,
            output_base_dir=config.output_base_dir,
            config=config
        )
        
        if config.resume:
            instance.load_state()
        
        await instance.get_steered_systems()
        return instance

    def load_state(self) -> None:
        self.steered_states = self._load_steered_system_states()
        self.responses = self._load_responses()
        self.scores = self._load_scores()

    async def get_steered_systems(self) -> None:
        if self.tested_system.supports_async_steering():
            await self._get_steered_systems_async()
        else:
            self._get_steered_systems_sync()

    def _get_steered_systems_sync(self) -> None:
        for persona in self.personas:
            if persona.persona_id in self.steered_states:
                # Restore from saved state
                state = self._restore_steered_system_from_state_sync(persona.persona_id)
                self.steered_systems[persona.persona_id] = state
            else:
                # Create new steered system
                system = self.tested_system.steer(persona,
                                                  self.steer_set.get_observations_by_persona(persona))
                self._save_steered_system_state(persona.persona_id, system.get_state())
                self.steered_systems[persona.persona_id] = system

    async def _get_steered_systems_async(self) -> None:
        tasks = []
        for persona in self.personas:
            if persona.persona_id in self.steered_states:
                # Restore from saved state
                if self.verbose:
                    print(f'Restoring steered system for {persona.persona_description}')
                tasks.append(self._restore_steered_system_from_state_async(persona.persona_id))
            else:
                # Create new steered system
                if self.verbose:
                    print(f'Creating new steered system for {persona.persona_description}')
                tasks.append(self._create_steered_system_async(persona))
        
        if tasks:
            steered_systems = await asyncio.gather(*tasks)
            for persona, system in zip(self.personas, steered_systems):
                if persona.persona_id not in self.steered_states:
                    if self.verbose:
                        print(f'Saving state for {persona.persona_description}')
                    self._save_steered_system_state(persona.persona_id, system.get_state())
                self.steered_systems[persona.persona_id] = system

    async def _create_steered_system_async(self, persona: Persona) -> BaseSteeredSystem:
        return await self.tested_system.steer_async(persona,
                                                    self.steer_set.get_observations_by_persona(persona))

    def _restore_steered_system_from_state_sync(self, persona_id: PersonaId) -> BaseSteeredSystem:
        state_data = self.steered_states[persona_id]
        state_class = self.tested_system.get_steered_state_class()
        state = state_class.from_dict(state_data)
        steered_system = self.tested_system.create_steered_from_state(state)
        return steered_system

    async def _restore_steered_system_from_state_async(self, persona_id: PersonaId) -> BaseSteeredSystem:
        state_data = self.steered_states[persona_id]
        state_class = self.tested_system.get_steered_state_class()
        state = state_class.from_dict(state_data)
        steered_system = await self.tested_system.create_steered_from_state_async(state)
        return steered_system


    async def run_eval(self, max_concurrent_tests: int = MAX_CONCURRENT_TESTS) -> None:
        semaphore = asyncio.Semaphore(max_concurrent_tests)
        tasks = []
        for steered_persona in self.personas:
            steered_system = self.steered_systems[steered_persona.persona_id]
            for test_persona in self.personas:
                if not self.has_score(steered_persona, test_persona):
                    tasks.append(self.test_steered_system_on_persona_async(
                        steered_system, test_persona, semaphore
                    ))
        await asyncio.gather(*tasks)

    async def test_steered_system_on_persona_async(
        self,
        steered_system: BaseSteeredSystem,
        test_persona: Persona,
        semaphore: asyncio.Semaphore
    ) -> float:
        test_observations = self.test_set.get_observations_by_persona(test_persona)
        test_persona_id = test_persona.persona_id
        steered_persona = steered_system.persona
        steered_persona_id = steered_persona.persona_id
        correct_responses = 0
        total_observations = min(len(test_observations), self.max_observations)
        sleep_time = 1
        n_errors = 0

        async with semaphore:
            responses_dict: Dict[str, Dict[str, str]] = {}
            if self.verbose:
                print(f'Testing {steered_persona.persona_description} on {test_persona.persona_description}')
            for test_observation in test_observations[:self.max_observations]:
                if self.has_response(steered_persona, test_persona, test_observation):
                    if self.verbose:
                        print(f'Skipping {steered_persona.persona_description} on {test_persona.persona_description} - {test_observation.observation_id}')
                    response_dict = self.responses[steered_persona_id][test_persona_id][test_observation.observation_id]
                    if response_dict['response'] == response_dict['correct_response']:
                        correct_responses += 1
                    continue
                correct_response = test_observation.correct_response
                have_response = False
                while not have_response:
                    try:
                        response = await steered_system.run_inference_async(test_observation)
                        have_response = True
                    except Exception as e:
                        sleep_time = sleep_time * 2 ** n_errors
                        print(f'Error running inference for {steered_persona.persona_description} on {test_persona.persona_description} - {test_observation.observation_id}. Sleeping for {sleep_time} seconds.')
                        print(e)
                        await asyncio.sleep(sleep_time)
                        n_errors += 1
                response_dict = {
                    'response': response,
                    'correct_response': correct_response
                }
                responses_dict[test_observation.observation_id] = response_dict
                if response == correct_response:
                    correct_responses += 1
        score = correct_responses / total_observations
        self._save_responses(steered_persona_id, test_persona_id, responses_dict)
        self._save_score(steered_persona_id, test_persona_id, score)
        return score

    def _get_responses_path(self) -> Path:
        """Get path to responses file"""
        return Path(self.output_dir) / f'responses_{self.experiment_name}.json'
    
    def _get_scores_path(self) -> Path:
        """Get path to scores file"""
        return Path(self.output_dir) / f'scores_{self.experiment_name}.json'

    def _load_responses(self) -> Dict:
        """Load saved responses if they exist"""
        path = self._get_responses_path()
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _load_scores(self) -> Dict:
        """Load saved scores if they exist"""
        path = self._get_scores_path()
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_responses(
        self,
        steered_id: str,
        test_id: str,
        responses_dict: Dict[str, Dict]
    ) -> None:
        if steered_id not in self.responses:
            self.responses[steered_id] = {}
        
        self.responses[steered_id][test_id] = responses_dict
        
        with open(self._get_responses_path(), 'w') as f:
            json.dump(self.responses, f)

    def _save_score(self, steered_id: str, test_id: str, score: float) -> None:
        if steered_id not in self.scores:
            self.scores[steered_id] = {}
        
        self.scores[steered_id][test_id] = score
        
        with open(self._get_scores_path(), 'w') as f:
            json.dump(self.scores, f)

    def has_score(self, steered_persona: Persona, test_persona: Persona) -> bool:
        return (steered_persona.persona_id in self.scores and
                test_persona.persona_id in self.scores[steered_persona.persona_id])

    def has_response(self, steered_persona: Persona, test_persona: Persona, observation: Observation) -> bool:
        return (steered_persona.persona_id in self.responses and
                test_persona.persona_id in self.responses[steered_persona.persona_id] and
                observation.observation_id in self.responses[steered_persona.persona_id][test_persona.persona_id])

    def _get_steered_states_path(self) -> Path:
        return Path(self.output_dir) / f'steered_states_{self.experiment_name}.json'

    def _load_steered_system_states(self) -> Dict[PersonaId, Dict[str, Any]]:
        path = self._get_steered_states_path()
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_steered_system_state(self, persona_id: PersonaId, state: SteeredSystemState) -> None:
        self.steered_states[persona_id] = state.to_dict()
        with open(self._get_steered_states_path(), 'w') as f:
            json.dump(self.steered_states, f)