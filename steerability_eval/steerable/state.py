from typing import Dict, Any, Type, ClassVar
from abc import ABC, abstractmethod

from steerability_eval.dataset.base import PersonaId

class SteeredSystemState(ABC):
    """Base class for steered system state"""
    
    state_type: ClassVar[str]  # Class variable identifying the state type
    
    def __init__(self, persona_id: PersonaId):
        self.persona_id = persona_id
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary"""
        return {"persona_id": self.persona_id}
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SteeredSystemState':
        """Create state instance from dictionary"""
        raise NotImplementedError
    
    @classmethod
    def get_state_type(cls) -> str:
        """Return string identifier for this state type"""
        return cls.state_type 