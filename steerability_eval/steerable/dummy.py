import random
from typing import List

from steerability_eval.steerable.base import BaseSteerableSystem, BaseSteeredSystem
from steerability_eval.dataset import Persona, Observation, SystemResponse

        
class DummySteerableSystem(BaseSteerableSystem):
    def __init__(self):
        pass

    def steer(self, persona: Persona, steer_observations: List[Observation]) -> BaseSteeredSystem:
        print(f'Steering to persona {persona.persona_id}')
        return DummySteeredSystem(persona, self, steer_observations)
    

class DummySteeredSystem(BaseSteeredSystem):
    def __init__(self, persona: Persona, steerable_system: BaseSteerableSystem, steer_observations: List[Observation]):
        super().__init__(persona, steerable_system, steer_observations)

    def run_inference(self, scenario) -> SystemResponse:
        # Random Y/N
        return 'Y' if random.random() < 0.5 else 'N'

    def __repr__(self):
        return f'DummySteeredSystem(persona={self.persona.persona_id}, steerable_system={self.steerable_system})'
