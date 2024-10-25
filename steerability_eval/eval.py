import asyncio
from datetime import datetime

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from steerability_eval.util import tqdm
from steerability_eval.steerable.base import BaseSteerableSystem, BaseSteeredSystem
from steerability_eval.dataset.w5 import W5Dataset, Persona, Observation


AGREE_STR = 'Y'
DISAGREE_STR = 'N'

DEFAULT_OUTPUT_FOLDER = 'output/images/'

MAX_CONCURRENT_TESTS = 8


class SteerabilityEval:
    def __init__(self, 
                 tested_system: BaseSteerableSystem, 
                 dataset: W5Dataset,
                 n_steer_observations_per_persona: int = 10,
                 random_state: int = 42):
        self.tested_system = tested_system
        self.dataset = dataset
        self.personas = self.dataset.personas
        self.steer_set, self.test_set = self.dataset.split(n_steer_observations_per_persona, random_state)
        self.max_observations = 100

    def steer_to_persona(self, persona: Persona) -> BaseSteeredSystem:
        steer_observations = self.steer_set.get_observations_by_persona(persona)
        steered_system = self.tested_system.steer(persona, steer_observations)
        return steered_system

    def test_steered_system_on_persona(self, steered_system: BaseSteeredSystem, test_persona: Persona) -> float:
        test_observations = self.test_set.get_observations_by_persona(test_persona)
        correct_responses = 0
        for i, test_observation in enumerate(test_observations[:self.max_observations]):
            response = steered_system.run_inference(test_observation)
            if response == AGREE_STR:
                correct_responses += 1
        return correct_responses / (i + 1)

    def run_eval(self):
        # Steer systems
        self.steered_systems = {}
        for persona in self.dataset.personas:
            steered_system = self.steer_to_persona(persona)
            self.steered_systems[persona.persona_id] = steered_system

        # Test every system on every persona
        self.steered_system_scores = {}
        for steered_system in tqdm(self.steered_systems.values(), desc='Testing steered systems'):
            self.steered_system_scores[steered_system.persona.persona_id] = {}
            for test_persona in tqdm(self.dataset.personas, desc='Testing steered system on personas'):
                print(f'Testing system {steered_system.persona.persona_description} on persona {test_persona.persona_description}')
                score = self.test_steered_system_on_persona(steered_system, test_persona)
                steered_key = f'steered_{steered_system.persona.persona_id}'
                test_key = f'test_{test_persona.persona_id}'
                self.steered_system_scores[steered_key][test_key] = score

        self.heatmap_fig = self.generate_heatmap()

    async def test_steered_system_on_persona_async(
        self,
        steered_system: BaseSteeredSystem,
        test_persona: Persona,
        semaphore: asyncio.Semaphore
    ) -> float:
        async with semaphore:
            print(f'Testing system {steered_system.persona.persona_description} on persona {test_persona.persona_description}')
            test_observations = self.test_set.get_observations_by_persona(test_persona)
            correct_responses = 0
            for i, test_observation in enumerate(test_observations[:self.max_observations]):
                response = await steered_system.run_inference_async(test_observation)
                if response == AGREE_STR:
                    correct_responses += 1
            score = correct_responses / (i + 1)
            self.steered_system_scores[steered_system.persona.persona_id][test_persona.persona_id] = score
            return score

    async def run_eval_async(self, max_concurrent_tests: int = MAX_CONCURRENT_TESTS):
        semaphore = asyncio.Semaphore(max_concurrent_tests)
        # Steer systems
        self.steered_systems = {}
        for persona in self.dataset.personas:
            steered_system = self.steer_to_persona(persona)
            self.steered_systems[persona.persona_id] = steered_system

        # Test every system on every persona
        tasks = []
        self.steered_system_scores = {}
        for steered_system in self.steered_systems.values():
            self.steered_system_scores[steered_system.persona.persona_id] = {}
            for test_persona in self.dataset.personas:
                tasks.append(self.test_steered_system_on_persona_async(
                    steered_system, test_persona, semaphore))
        return await asyncio.gather(*tasks)

    def generate_heatmap(self) -> plt.Figure:
        scores_df = pd.DataFrame(self.steered_system_scores)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(scores_df, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax)
        ax.set_xticklabels([p.persona_description for p in self.dataset.personas], rotation=45, ha='right')
        ax.set_yticklabels([p.persona_description for p in self.dataset.personas], rotation=0)
        ax.set_title('Steerability Eval Heatmap')
        ax.set_xlabel('Test Persona')
        ax.set_ylabel('Steered System Persona')
        # x axis on top instead of bottom
        ax.xaxis.tick_top()
        plt.tight_layout()
        return fig

    def save_heatmap(self, output_folder: str):
        self.heatmap_fig.savefig(
            f'{output_folder}/heatmap_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
        )
