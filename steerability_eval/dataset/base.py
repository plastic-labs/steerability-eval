from dataclasses import dataclass
from typing import Optional, List, Tuple, Type

import pandas as pd


SystemResponse = str
PersonaId = str
ScenarioId = str
ObservationId = str


@dataclass
class Persona:
    persona_id: PersonaId
    persona_description: str
    framework: Optional[str]

    @classmethod
    def from_row(cls, row: pd.Series):
        return cls(
            persona_id=row['persona_id'],
            persona_description=row['persona_description'],
            framework=row['framework']
        )

    def __repr__(self):
        return f'Persona(persona_id={self.persona_id}, persona_description={self.persona_description}, framework={self.framework})'


@dataclass
class Observation:
    observation_id: ObservationId
    response: str
    scenario_id: ScenarioId
    scenario: str
    persona_id: PersonaId
    correct_response: str
    
    @classmethod
    def from_row(cls, row: pd.Series):
        return cls(
            observation_id=row['observation_id'],
            response=row['response'],
            scenario_id=row['scenario_id'],
            scenario=row['scenario'],
            persona_id=row['persona_id'],
            correct_response=row['correct_response']
        )

    def __repr__(self):
        return f'Observation(observation_id={self.observation_id}, persona_id={self.persona_id}, scenario={self.scenario[:30]}, response={self.response[:30]})'

@dataclass
class BaseDataset:
    personas_df: pd.DataFrame
    observations_df: pd.DataFrame

    @classmethod
    def merge(cls, datasets: List['BaseDataset']) -> 'BaseDataset':
        dataset_class = datasets[0].__class__
        assert all([d.__class__ == dataset_class for d in datasets]), 'All datasets must be of the same class'
        return dataset_class(
            personas_df=pd.concat([d.personas_df for d in datasets], ignore_index=True),
            observations_df=pd.concat([d.observations_df for d in datasets]),
            max_personas=sum([d.max_personas for d in datasets])
        )

    @property
    def personas(self) -> List[Persona]:
        return [Persona.from_row(row) for _, row in self.personas_df.iterrows()]

    def split(self, n_steer_observations_per_persona: int) -> Tuple['BaseDataset', 'BaseDataset']:
        raise NotImplementedError

    def get_observations_by_persona(self, persona: Persona) -> List[Observation]:
        raise NotImplementedError