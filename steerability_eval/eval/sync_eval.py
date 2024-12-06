from typing import Dict, Optional, Any
from pathlib import Path

from steerability_eval.eval.base import (
    BaseEval, BaseSteerableSystem, BaseSteeredSystem, Persona, MAX_CONCURRENT_TESTS, MAX_OBSERVATIONS
)
from steerability_eval.dataset.base import BaseDataset, Observation, PersonaId
from steerability_eval.steerable.state import SteeredSystemState
from steerability_eval.eval.config import EvalConfig


class SteerabilityEval(BaseEval):
    """Synchronous evaluation implementation"""
    def __init__(self, 
                 tested_system: BaseSteerableSystem, 
                 dataset: BaseDataset,
                 experiment_name: Optional[str] = None,
                 n_steer_observations_per_persona: int = 5,
                 max_observations: int = MAX_OBSERVATIONS,
                 verbose: bool = False,
                 output_base_dir: str = 'output/experiments'):
        super().__init__(
            tested_system=tested_system,
            dataset=dataset,
            experiment_name=experiment_name,
            max_observations=max_observations,
            n_steer_observations_per_persona=n_steer_observations_per_persona,
            verbose=verbose,
            output_base_dir=output_base_dir
        )

    @classmethod
    def create(
        cls,
        tested_system: BaseSteerableSystem,
        dataset: BaseDataset,
        config: EvalConfig,
    ) -> 'SteerabilityEval':
        instance = cls(
            tested_system=tested_system,
            dataset=dataset,
            experiment_name=config.experiment_name,
            n_steer_observations_per_persona=config.n_steer_observations_per_persona,
            max_observations=config.max_observations,
            verbose=config.verbose,
            output_base_dir=config.output_base_dir
        )
        
        if config.resume:
            instance.load_state()
        
        instance.get_steered_systems()
        return instance

    def load_state(self) -> None:
        """Load saved state if it exists"""
        self.steered_states = self._load_steered_system_states()
        self.responses = self._load_responses()
        self.scores = self._load_scores()

    def get_steered_systems(self) -> None:
        """Get steered systems"""
        for persona in self.personas:
            if persona.persona_id in self.steered_states:
                # Restore from saved state
                state = self._restore_steered_system_from_state(persona.persona_id)
                self.steered_systems[persona.persona_id] = state
            else:
                # Create new steered system
                system = self.tested_system.steer(persona,
                                                self.steer_set.get_observations_by_persona(persona))
                self._save_steered_system_state(persona.persona_id, system.get_state())
                self.steered_systems[persona.persona_id] = system

    def _restore_steered_system_from_state(self, persona_id: PersonaId) -> BaseSteeredSystem:
        """Restore a steered system from saved state"""
        state_data = self.steered_states[persona_id]
        state_class = self.tested_system.get_steered_state_class()
        state = state_class.from_dict(state_data)
        return self.tested_system.create_steered_from_state(state)

    def run_eval(self) -> None:
        """Run evaluation synchronously"""
        for steered_persona in self.personas:
            steered_system = self.steered_systems[steered_persona.persona_id]
            for test_persona in self.personas:
                if not self.has_score(steered_persona, test_persona):
                    self.test_steered_system_on_persona(steered_system, test_persona)

    def test_steered_system_on_persona(
        self,
        steered_system: BaseSteeredSystem,
        test_persona: Persona,
    ) -> float:
        """Test a steered system on a persona"""
        test_observations = self.test_set.get_observations_by_persona(test_persona)
        test_persona_id = test_persona.persona_id
        steered_persona = steered_system.persona
        steered_persona_id = steered_persona.persona_id
        correct_responses = 0
        total_observations = min(len(test_observations), self.max_observations)

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
            response = steered_system.run_inference(test_observation)
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