from dataclasses import dataclass
from typing import Optional

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

    @classmethod
    def from_row(cls, row: pd.Series):
        return cls(
            observation_id=row['observation_id'],
            response=row['response'],
            scenario_id=row['scenario_id'],
            scenario=row['scenario'],
            persona_id=row['persona_id']
        )

    def __repr__(self):
        return f'Observation(observation_id={self.observation_id}, persona_id={self.persona_id}, scenario={self.scenario[:30]}, response={self.response[:30]})'
