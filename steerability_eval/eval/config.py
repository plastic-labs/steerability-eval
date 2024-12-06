from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class EvalConfig:
    """Configuration for evaluation runs"""
    
    # Steerable system settings
    steerable_system_type: str = ''
    steerable_system_config: Dict[str, Any] = field(default_factory=dict)
    
    # Dataset settings
    personas_path: str = ''
    observations_path: str = ''
    max_personas: int = 0  # 0 means use all personas
    random_state: int = 42
    n_steer_observations_per_persona: int = 4
    max_observations: int = 100  # Max observations to test per persona
    
    # Runtime settings
    run_async: bool = False
    restore_async: bool = False
    resume: bool = False
    verbose: bool = False
    max_concurrent_tests: int = 8
    output_base_dir: str = 'output/experiments'
    experiment_name: Optional[str] = None
    inference_batch_size: int = 10

    def to_dict(self) -> dict:
        """Convert config to dictionary for saving"""
        return {
            'personas_path': self.personas_path,
            'observations_path': self.observations_path,
            'max_personas': self.max_personas,
            'random_state': self.random_state,
            'n_steer_observations_per_persona': self.n_steer_observations_per_persona,
            'max_observations': self.max_observations,
            'run_async': self.run_async,
            'restore_async': self.restore_async,
            'resume': self.resume,
            'verbose': self.verbose,
            'max_concurrent_tests': self.max_concurrent_tests,
            'output_base_dir': self.output_base_dir,
            'experiment_name': self.experiment_name,
            'steerable_system_type': self.steerable_system_type,
            'steerable_system_config': self.steerable_system_config,
            'inference_batch_size': self.inference_batch_size
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EvalConfig':
        """Create config from dictionary"""
        return cls(**data) 