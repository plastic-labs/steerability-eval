from typing import List, Type, Optional
from abc import abstractmethod

from steerability_eval.dataset.base import Persona, Observation, SystemResponse, PersonaId
from steerability_eval.steerable.state import SteeredSystemState

class BaseSteerableSystem:
    def __init__(self):
        pass

    def steer(self, persona: Persona, steer_observations: List[Observation]) -> 'BaseSteeredSystem':
        raise NotImplementedError
    
    async def steer_async(self, persona: Persona, steer_observations: List[Observation]) -> 'BaseSteeredSystem':
        """Optional async steering implementation"""
        raise NotImplementedError
    
    @staticmethod
    def supports_async_steering() -> bool:
        """Whether this system supports async steering"""
        raise NotImplementedError
    
    @staticmethod
    def supports_saving_state() -> bool:
        """Whether this system supports saving state"""
        raise NotImplementedError
    
    @abstractmethod
    def get_steered_state_class(self) -> Type[SteeredSystemState]:
        """Returns the state class used by this system's steered instances"""
        raise NotImplementedError
    
    @abstractmethod
    def create_steered_from_state(self, state: SteeredSystemState) -> 'BaseSteeredSystem':
        """Create a steered system instance from a state"""
        raise NotImplementedError

    async def create_steered_from_state_async(self, state: SteeredSystemState) -> 'BaseSteeredSystem':
        """Create a steered system instance from a state"""
        raise NotImplementedError

    @staticmethod
    def supports_batch_inference() -> bool:
        """Whether this system supports batch inference"""
        raise NotImplementedError

    @staticmethod
    def supports_async_inference() -> bool:
        """Whether this system supports async inference"""
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

    async def run_inference_async(self, observation: Observation) -> SystemResponse:
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self) -> SteeredSystemState:
        """Get serializable state for this steered system"""
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(persona={self.persona.persona_id}, steerable_system={self.steerable_system})'

    def wait_until_ready(self) -> None:
        """Wait for system to be ready for inference"""
        pass


    async def run_batch_inference_async(
        self, 
        observations: List[Observation]
    ) -> List[SystemResponse]:
        """Run inference on multiple observations at once"""
        raise NotImplementedError
