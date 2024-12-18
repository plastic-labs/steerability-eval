import asyncio
import time
from datetime import datetime
import os
import json
from typing import Dict, List, Optional, Union, Any, Type
from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from steerability_eval.util import tqdm
from steerability_eval.steerable.base import BaseSteerableSystem, BaseSteeredSystem
from steerability_eval.dataset.base import BaseDataset, Persona, Observation, PersonaId
from steerability_eval.dataset import get_dataset_class
from steerability_eval.steerable import get_steerable_system_class
from steerability_eval.steerable.state import SteeredSystemState

AGREE_STR = 'Y'
DISAGREE_STR = 'N'
DEFAULT_OUTPUT_FOLDER = 'output/images/'
MAX_CONCURRENT_TESTS = 8
MAX_OBSERVATIONS = 100


class BaseEval(ABC):
    """Base class for evaluation implementations"""
    def __init__(self, 
                 tested_system: BaseSteerableSystem, 
                 dataset: BaseDataset,
                 experiment_name: Optional[str] = None,
                 n_steer_observations_per_persona: int = 5,
                 verbose: bool = False,
                 output_base_dir: str = 'output/experiments'):
        self.tested_system = tested_system
        self.dataset = dataset
        self.personas = self.dataset.personas
        self.n_steer_observations_per_persona = n_steer_observations_per_persona
        self.steer_set, self.test_set = self.dataset.split(n_steer_observations_per_persona)
        self.output_base_dir = output_base_dir
        self.max_observations = MAX_OBSERVATIONS
        self.verbose = verbose

        # Set up experiment directory
        self.experiment_name = experiment_name or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_dir = os.path.join(output_base_dir, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize state storage
        self.steered_systems: Dict[PersonaId, BaseSteeredSystem] = {}
        self.steered_states: Dict[PersonaId, Dict[str, Any]] = self._load_states()
        self.responses: Dict = self._load_responses()
        self.steered_system_scores: Dict = self._load_scores()

    # Common methods for both sync/async implementations
    def _get_states_path(self) -> str:
        return os.path.join(self.output_dir, f'states_{self.experiment_name}.json')

    def _load_states(self) -> Dict[PersonaId, Dict[str, Any]]:
        path = self._get_states_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self, persona_id: PersonaId, state: SteeredSystemState) -> None:
        """Save state for a single steered system"""
        self.steered_states[persona_id] = {
            'state_type': state.get_state_type(),
            'state_data': state.to_dict()
        }
        with open(self._get_states_path(), 'w') as f:
            json.dump(self.steered_states, f)

    def _get_responses_path(self) -> str:
        return os.path.join(self.output_dir, f'responses_{self.experiment_name}.json')
    
    def _get_scores_path(self) -> str:
        return os.path.join(self.output_dir, f'scores_{self.experiment_name}.json')
    
    def _get_params_path(self) -> str:
        return os.path.join(self.output_dir, f'params_{self.experiment_name}.json')

    def _load_responses(self) -> Dict:
        path = self._get_responses_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _load_scores(self) -> Dict:
        path = self._get_scores_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _load_params(self) -> None:
        path = self._get_params_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                params = json.load(f)
                self.n_steer_observations_per_persona = params['n_steer_observations_per_persona']
                # Load other params as needed

    def save_params(self) -> None:
        params = {
            'n_steer_observations_per_persona': self.n_steer_observations_per_persona,
            'n_personas': len(self.personas),
            'llm_provider': getattr(self.tested_system, 'llm_provider', None),
            'experiment_name': self.experiment_name,
            'max_observations': self.max_observations,
            'personas_path': self.dataset.personas_path,
            'observations_path': self.dataset.observations_path,
            'steerable_system_class': self.tested_system.__class__.__name__,
            'dataset_class': self.dataset.__class__.__name__,
            'output_base_dir': self.output_base_dir,
            'verbose': self.verbose,
            'experiment_name': self.experiment_name,
            'max_personas': self.dataset.max_personas,
            'random_state': self.dataset.random_state
        }
        
        with open(self._get_params_path(), 'w') as f:
            json.dump(params, f)

    def _save_responses(
        self,
        steered_id: str,
        test_id: str,
        responses_dict: Dict[str, Dict]
    ) -> None:
        if steered_id not in self.responses:
            self.responses[steered_id] = {}
        
        self.responses[steered_id][test_id] = responses_dict
        
        # Save to file
        with open(self._get_responses_path(), 'w') as f:
            json.dump(self.responses, f)

    def _save_score(self, steered_id: str, test_id: str, score: float) -> None:
        if steered_id not in self.steered_system_scores:
            self.steered_system_scores[steered_id] = {}
        
        self.steered_system_scores[steered_id][test_id] = score
        
        # Save to file
        with open(self._get_scores_path(), 'w') as f:
            json.dump(self.steered_system_scores, f)

    def has_score(self, steered_persona: Persona, test_persona: Persona) -> bool:
        return (steered_persona.persona_id in self.steered_system_scores and
                test_persona.persona_id in self.steered_system_scores[steered_persona.persona_id])

    def has_response(self, steered_persona: Persona, test_persona: Persona, observation: Observation) -> bool:
        return (steered_persona.persona_id in self.responses and
                test_persona.persona_id in self.responses[steered_persona.persona_id] and
                observation.observation_id in self.responses[steered_persona.persona_id][test_persona.persona_id])

    def get_responses_df(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for steered_id, test_personas in self.responses.items():
            for test_id, observations in test_personas.items():
                for obs_id, results in observations.items():
                    rows.append({   
                        'steered_persona_id': steered_id,
                        'test_persona_id': test_id,
                        'observation_id': obs_id,
                        'system_response': results['response'],
                        'correct_response': results['correct_response']
                    })
        df = pd.DataFrame(rows)
        df['is_correct'] = df['system_response'] == df['correct_response']
        return df

    def generate_heatmap(self) -> Figure:
        scores_df = pd.DataFrame(self.steered_system_scores).sort_index(axis=1).sort_index(axis=0)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(scores_df, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax)
        ax.set_xticklabels([p.persona_description for p in self.dataset.personas], rotation=45, ha='right')
        ax.set_yticklabels([p.persona_description for p in self.dataset.personas], rotation=0)
        ax.set_title('Steerability Eval Heatmap')
        ax.set_ylabel('Test Persona')
        ax.set_xlabel('Steered System Persona')
        ax.xaxis.tick_top()
        plt.tight_layout()
        return fig

    def save_heatmap(self, output_folder: Optional[str] = None) -> None:
        output_folder = output_folder or self.output_dir
        fig = self.generate_heatmap()
        fig.savefig(os.path.join(output_folder, f'heatmap_{self.experiment_name}.png'))

    @abstractmethod
    def initialize(self) -> None:
        """Initialize steered systems"""
        pass

    @abstractmethod
    def run_eval(self) -> None:
        """Run evaluation"""
        pass


class SteerabilityEval(BaseEval):
    """Synchronous evaluation implementation"""
    def initialize(self) -> None:
        """Initialize steered systems synchronously"""
        for persona in self.personas:
            if persona.persona_id in self.steered_states:
                # Restore from saved state
                state_data = self.steered_states[persona.persona_id]
                state_class = self.tested_system.get_steered_state_class()
                state = state_class.from_dict(state_data['state_data'])
                self.steered_systems[persona.persona_id] = self.tested_system.create_steered_from_state(state)
            else:
                # Create new steered system
                steered_system = self.steer_to_persona(persona)
                self._save_state(persona.persona_id, steered_system.get_state())
                self.steered_systems[persona.persona_id] = steered_system

    def steer_to_persona(self, persona: Persona) -> BaseSteeredSystem:
        steer_observations = self.steer_set.get_observations_by_persona(persona)
        steered_system = self.tested_system.steer(persona, steer_observations)
        return steered_system

    def run_eval(self) -> None:
        # Save initial params
        self.save_params()
        
        # Test every system on every persona
        for steered_persona in self.personas:
            steered_system = self.steered_systems[steered_persona.persona_id]
            for test_persona in self.personas:
                if not self.has_score(steered_persona, test_persona):
                    self.test_steered_system_on_persona(steered_system, test_persona)

    def test_steered_system_on_persona(self, steered_system: BaseSteeredSystem, test_persona: Persona) -> float:
        test_observations = self.test_set.get_observations_by_persona(test_persona)
        test_persona_id = test_persona.persona_id
        steered_persona = steered_system.persona
        steered_persona_id = steered_system.persona.persona_id
        correct_responses = 0
        total_observations = min(len(test_observations), self.max_observations)
        responses_dict: Dict[str, Dict[str, str]] = {}
        sleep_time = 1
        n_errors = 0

        if self.verbose:
            print(f'Testing {steered_persona.persona_description} on {test_persona.persona_description}')
        for test_observation in test_observations[:self.max_observations]:
            observation_id = test_observation.observation_id
            # Skip if we already have this response
            if self.has_response(steered_persona, test_persona, test_observation):
                if self.verbose:
                    print(f'Skipping {steered_persona.persona_description} on {test_persona.persona_description} - {observation_id}')
                response_dict = self.responses[steered_persona_id][test_persona_id][observation_id]
                if response_dict['response'] == response_dict['correct_response']:
                    correct_responses += 1
                continue
            correct_response = test_observation.correct_response
            have_response = False
            while not have_response:
                try:
                    response = steered_system.run_inference(test_observation)
                    have_response = True
                except Exception as e:
                    sleep_time = sleep_time * 2 ** n_errors
                    print(f'Error running inference for {steered_persona.persona_description} on {test_persona.persona_description} - {test_observation.observation_id}. Sleeping for {sleep_time} seconds.')
                    print(e)
                    time.sleep(sleep_time)
                    n_errors += 1
            response_dict = {
                'response': response,
                'correct_response': correct_response
            }
            responses_dict[observation_id] = response_dict
            if response == correct_response:
                correct_responses += 1

        self._save_responses(steered_persona_id, test_persona_id, responses_dict)
        score = correct_responses / total_observations
        self._save_score(steered_persona_id, test_persona_id, score)
        return score


class AsyncSteerabilityEval(BaseEval):
    """Asynchronous evaluation implementation"""
    async def initialize(self) -> None:
        """Initialize steered systems asynchronously"""
        tasks = []
        for persona in self.personas:
            if persona.persona_id in self.steered_states:
                # Restore from saved state
                state_data = self.steered_states[persona.persona_id]
                state_class = self.tested_system.get_steered_state_class()
                state = state_class.from_dict(state_data['state_data'])
                self.steered_systems[persona.persona_id] = self.tested_system.create_steered_from_state(state)
            else:
                # Create new steered system
                tasks.append(self.steer_to_persona_async(persona))
        
        if tasks:
            steered_systems = await asyncio.gather(*tasks)
            for persona, system in zip(self.personas, steered_systems):
                if persona.persona_id not in self.steered_states:
                    self._save_state(persona.persona_id, system.get_state())
                    self.steered_systems[persona.persona_id] = system

    async def steer_to_persona_async(self, persona: Persona) -> BaseSteeredSystem:
        steer_observations = self.steer_set.get_observations_by_persona(persona)
        steered_system = await self.tested_system.steer_async(persona, steer_observations)
        return steered_system

    async def run_eval(self, max_concurrent_tests: int = MAX_CONCURRENT_TESTS) -> None:
        # Save initial params
        self.save_params()
        
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
                        print(e.__traceback__)
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

    @classmethod
    def from_existing(cls, params_path: str) -> 'SteerabilityEval':
        # Load params
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Create dataset
        dataset_class = get_dataset_class(params['dataset_class'])
        dataset = dataset_class.from_csv(
            personas_path=params['personas_path'],
            observations_path=params['observations_path'],
            max_personas=params['max_personas'],
            random_state=params['random_state']
        ) # type: ignore

        # Create steerable system
        steerable_system_class = get_steerable_system_class(params['steerable_system_class'])
        tested_system = steerable_system_class()
        
        # Create eval
        instance = cls(
            tested_system=tested_system,
            dataset=dataset,
            n_steer_observations_per_persona=int(params['n_steer_observations_per_persona']),
            output_base_dir=params['output_base_dir'],
            verbose=params['verbose'],
            experiment_name=params['experiment_name']
        )
        
        return instance

    @classmethod
    async def create_async(cls, *args, **kwargs) -> 'SteerabilityEval':
        kwargs['steer_async'] = True
        instance = cls(*args, **kwargs)
        await instance.initialize()
        return instance

    @classmethod
    def create(cls, *args, **kwargs) -> 'SteerabilityEval':
        kwargs['steer_async'] = False
        instance = cls(*args, **kwargs)
        instance.initialize()
        print(instance.steered_systems)
        return instance