import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from steerability_eval.steerable.base import BaseSteerableSystem, BaseSteeredSystem
from steerability_eval.dataset.w5 import W5Dataset, Persona, Observation


AGREE_STR = 'Y'
DISAGREE_STR = 'N'


class SteerabilityEval:
    def __init__(self, tested_system: BaseSteerableSystem, dataset: W5Dataset):
        self.tested_system = tested_system
        self.dataset = dataset
        self.personas = self.dataset.personas
        self.steer_set, self.test_set = self.dataset.split()
        self.max_observations = 100

    def steer_to_persona(self, persona: Persona) -> BaseSteeredSystem:
        steer_observations = self.steer_set.get_observations(persona)
        steered_system = self.tested_system.steer(persona, steer_observations)
        return steered_system

    def test_steered_system_on_persona(self, steered_system: BaseSteeredSystem, test_persona: Persona) -> float:
        test_observations = self.test_set.get_observations(test_persona)
        correct_responses = 0
        for i, test_observation in enumerate(test_observations[:self.max_observations]):
            response = steered_system.run_inference(test_observation)
            if response == AGREE_STR:
                correct_responses += 1
        return correct_responses / (i + 1)


    def run_eval(self):
        # Steer systems
        self.steered_systems = {}
        for persona in self.dataset.personas.values():
            steered_system = self.steer_to_persona(persona)
            self.steered_systems[persona.persona_id] = steered_system

        # Test every system on every persona
        self.steered_system_scores = {}
        for steered_system in self.steered_systems.values():
            self.steered_system_scores[steered_system.persona.persona_id] = {}
            for test_persona in self.dataset.personas.values():
                score = self.test_steered_system_on_persona(steered_system, test_persona)
                self.steered_system_scores[steered_system.persona.persona_id][test_persona.persona_id] = score

        self.heatmap_fig = self.generate_heatmap()
    
    def generate_heatmap(self):
        scores_df = pd.DataFrame(self.steered_system_scores)
        fig = sns.heatmap(scores_df, annot=True, fmt='.2f', cmap='viridis', cbar=True)
        fig.set_title('Steerability Eval Heatmap')
        fig.set_xlabel('Test Persona')
        fig.set_ylabel('Steered System Persona')
        return fig