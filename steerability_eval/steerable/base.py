from typing import List
from steerability_eval.dataset import Persona, Observation, SystemResponse


class BaseSteerableSystem:
    def __init__(self):
        pass

    def steer(self, persona: Persona, steer_observations: List[Observation]) -> 'BaseSteeredSystem':
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class BaseSteeredSystem:
    def __init__(self, persona: Persona, steerable_system: BaseSteerableSystem, steer_observations: List[Observation]):
        self.persona = persona
        self.steerable_system = steerable_system
        self.steer_observations = steer_observations

    def run_inference(self, observation: Observation) -> SystemResponse:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(persona={self.persona.persona_id}, steerable_system={self.steerable_system})'