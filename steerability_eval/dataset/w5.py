from typing import List, Dict, Set, Tuple

import pandas as pd

SystemResponse = str
PersonaId = str
ScenarioId = str
ObservationId = str
MAX_PERSONAS = 20
MAX_OBSERVATIONS_PER_PERSONA = 10

import hashlib

def generate_short_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]

class Persona:
    def __init__(self, persona_id: PersonaId, persona_description: str, persona_framework: str):
        self.persona_id = persona_id
        self.persona_description = persona_description
        self.persona_framework = persona_framework

    def __repr__(self):
        return f'Persona(persona_id={self.persona_id}, persona_description={self.persona_description}, persona_framework={self.persona_framework})'


class Observation:
    def __init__(
        self,
        observation_id: ObservationId,
        response: str,
        scenario_id: ScenarioId,
        scenario_description: str,
        persona_id: PersonaId,
    ):
        self.observation_id = observation_id
        self.response = response
        self.scenario_id = scenario_id
        self.scenario_description = scenario_description
        self.persona_id = persona_id
    def __repr__(self):
        return f'Observation(observation_id={self.observation_id}, persona_id={self.persona_id}, scenario_description={self.scenario_description[:30]}, response={self.response[:30]})'

class Dataset:
    def __init__(
        self,
        personas: Dict[PersonaId, Persona],
        observations: List[Observation],
        max_personas: int = MAX_PERSONAS,
        max_observations_per_persona: int = MAX_OBSERVATIONS_PER_PERSONA,
        response_types: List[str] = ['action'],
    ):
        self.max_personas = max_personas
        self.max_observations_per_persona = max_observations_per_persona
        self.response_types = response_types
        self.personas = personas
        self.observations = observations

    @classmethod
    def from_csv(cls,
                 personas_path: str,
                 observations_path: str,
                 max_personas: int = MAX_PERSONAS,
                 max_observations_per_persona: int = MAX_OBSERVATIONS_PER_PERSONA,
                 use_actions: bool = True,
                 use_thoughts: bool = False,
                 use_emotions: bool = False):
        personas = cls.load_personas(personas_path, max_personas)
        persona_ids = list(personas.keys())
        response_types = []
        if use_actions:
            response_types.append('action')
        if use_thoughts:
            response_types.append('thought')
        if use_emotions:
            response_types.append('emotion')
        observations = cls.load_observations(observations_path, response_types, persona_ids)
        return cls(personas,
                   observations,
                   max_personas,
                   max_observations_per_persona,
                   response_types)

    @classmethod
    def load_personas(cls, personas_path: str, max_personas: int = MAX_PERSONAS) -> Dict[PersonaId, Persona]:
        with open(personas_path, 'r') as f:
            df = pd.read_csv(f)
        df = df.head(max_personas)
        personas = {}
        for index, row in df.iterrows():
            persona = Persona(row['persona_id'], row['persona_description'], row['framework_name'])
            personas[persona.persona_id] = persona
        return personas

    @classmethod
    def load_observations(cls,
                          observations_path: str,
                          response_types: List[str],
                          persona_ids: List[PersonaId]) -> List[Observation]:
        with open(observations_path, 'r') as f:
            df = pd.read_csv(f)
        observations = []
        for index, row in df.iterrows():
            if row['persona_id'] not in persona_ids:
                continue
            scenario = f'{row["context"]}\n{row["scenario"]}'
            scenario_id = row['scenario_id']
            for response_type in response_types:
                response = row[response_type]
                observation_id = generate_short_hash(f'{response}{scenario_id}')
                observation = Observation(observation_id, response, scenario_id, scenario, row['persona_id'])
                observations.append(observation)
        return observations

    @property
    def persona_ids(self) -> Set[PersonaId]:
        return set(self.personas.keys())

    def get_observations(self, persona: Persona) -> List[Observation]:
        return [observation for observation in self.observations if observation.persona_id == persona.persona_id]

    def split(self, n_steer_observations_per_persona: int = 5) -> Tuple['Dataset', 'Dataset']:
        steer_observations = []
        test_observations = []
        for persona in self.personas.values():
            observations = self.get_observations(persona)
            steer_observations.extend(observations[:n_steer_observations_per_persona])
            test_observations.extend(observations[n_steer_observations_per_persona:])
        return Dataset(self.personas, steer_observations), Dataset(self.personas, test_observations)