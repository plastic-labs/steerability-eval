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
from steerability_eval.eval.config import EvalConfig

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
                 max_observations: int = MAX_OBSERVATIONS,
                 verbose: bool = False,
                 output_base_dir: str = 'output/experiments',
                 config: Optional[EvalConfig] = None):
        # Set up basic structure and paths
        self.tested_system = tested_system
        self.dataset = dataset
        self.personas = self.dataset.personas
        self.n_steer_observations_per_persona = n_steer_observations_per_persona
        self.max_observations = max_observations
        self.steer_set, self.test_set = self.dataset.split(n_steer_observations_per_persona)
        self.verbose = verbose
        self.config = config

        # Set up experiment directory
        self.experiment_name = experiment_name or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_dir = os.path.join(output_base_dir, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize empty containers
        self.steered_systems: Dict[PersonaId, BaseSteeredSystem] = {}
        self.steered_states: Dict[PersonaId, Dict[str, Any]] = {}
        self.responses: Dict = {}
        self.scores: Dict = {}

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
        """Save responses for a steered system tested on a persona"""
        if steered_id not in self.responses:
            self.responses[steered_id] = {}
        
        self.responses[steered_id][test_id] = responses_dict
        
        # Save to file
        with open(self._get_responses_path(), 'w') as f:
            json.dump(self.responses, f)

    def _save_score(self, steered_id: str, test_id: str, score: float) -> None:
        """Save score for a steered system tested on a persona"""
        if steered_id not in self.scores:
            self.scores[steered_id] = {}
        
        self.scores[steered_id][test_id] = score
        
        # Save to file
        with open(self._get_scores_path(), 'w') as f:
            json.dump(self.scores, f)

    def has_score(self, steered_persona: Persona, test_persona: Persona) -> bool:
        """Check if we have a score for this combination"""
        return (steered_persona.persona_id in self.scores and
                test_persona.persona_id in self.scores[steered_persona.persona_id])

    def has_response(self, steered_persona: Persona, test_persona: Persona, observation: Observation) -> bool:
        """Check if we have a response for this combination"""
        return (steered_persona.persona_id in self.responses and
                test_persona.persona_id in self.responses[steered_persona.persona_id] and
                observation.observation_id in self.responses[steered_persona.persona_id][test_persona.persona_id])
