from typing import List, Dict, Set, Tuple, Optional

import pandas as pd

from steerability_eval.dataset.base import BaseDataset, Persona, Observation, PersonaId, ObservationId, ScenarioId
from steerability_eval.util import generate_short_hash

MAX_PERSONAS = 20


class StatementsDataset(BaseDataset):
    def __init__(
        self,
        personas_df: pd.DataFrame,
        observations_df: pd.DataFrame,
        max_personas: int = MAX_PERSONAS,
        random_state: int = 42,
        personas_path: Optional[str] = None,
        observations_path: Optional[str] = None
    ):
        self.max_personas = max_personas
        self.personas_df = personas_df
        self.observations_df = observations_df
        self.random_state = random_state
        self.personas_path = personas_path
        self.observations_path = observations_path

    @classmethod
    def from_csv(cls,
                 personas_path: str,
                 observations_path: str,
                 max_personas: int = MAX_PERSONAS,
                 random_state: int = 42):
        personas_df = cls.load_personas(personas_path, max_personas, random_state)
        persona_ids = list(personas_df['persona_id'].unique())
        observations_df = cls.load_observations(
            observations_path,
            persona_ids,
            random_state
        )
        return cls(personas_df,
                   observations_df,
                   personas_path=personas_path,
                   observations_path=observations_path,
                   max_personas=max_personas,
                   random_state=random_state)

    @classmethod
    def load_personas(cls, personas_path: str, max_personas: int = MAX_PERSONAS, random_state: int = 42) -> pd.DataFrame:
        with open(personas_path, 'r') as f:
            df = pd.read_csv(f)
        n_personas = min(max_personas, len(df))
        if n_personas > 0:
            df = df.sample(n=n_personas, random_state=random_state, replace=False)
        df.rename(columns={'framework_name': 'framework'}, inplace=True)
        return df

    @classmethod
    def load_observations(cls,
                          observations_path: str,
                          persona_ids: List[PersonaId],
                          random_state: int = 42) -> pd.DataFrame:
        with open(observations_path, 'r') as f:
            df = pd.read_csv(f)
        df = df[df['persona_id'].isin(persona_ids)]
        df.rename(columns={'statement': 'response'}, inplace=True)
        df.rename(columns={'is_agree': 'correct_response'}, inplace=True)
        df['scenario'] = ''
        df['scenario_id'] = ''
        df['correct_response'] = df['correct_response'].map({True: 'Y', False: 'N'})
        df['observation_id'] = df.apply(
            lambda row: generate_short_hash( f"{row['persona_id']}_{row['response']}_{row['correct_response']}"),
            axis=1
        )
        return df

    def get_persona(self, persona_id: PersonaId) -> Persona:
        return Persona.from_row(self.personas_df[self.personas_df['persona_id'] == persona_id].iloc[0])

    def get_observation(self, observation_id: ObservationId) -> Observation:
        return Observation.from_row(self.observations_df[self.observations_df['observation_id'] == observation_id].iloc[0])

    def get_observations_by_persona(self, persona: Persona) -> List[Observation]:
        return [Observation.from_row(row) 
                for _, row in 
                self.observations_df[ self.observations_df['persona_id'] == persona.persona_id].iterrows()
                ]

    def split(self, n_steer_observations_per_persona: int = 10, random_state: int = 42) -> Tuple['StatementsDataset', 'StatementsDataset']:
        # Group by persona and correct_response to maintain balance
        steer_observations_df = pd.DataFrame()
        n_each = n_steer_observations_per_persona // 2  # Split evenly between Y/N
        
        for persona_id in self.observations_df['persona_id'].unique():
            persona_df = self.observations_df[self.observations_df['persona_id'] == persona_id]
            
            # Sample equal numbers of Y and N responses
            y_samples = persona_df[persona_df['correct_response'] == 'Y'].sample(
                n=n_each, random_state=random_state, replace=False
            )
            n_samples = persona_df[persona_df['correct_response'] == 'N'].sample(
                n=n_each, random_state=random_state, replace=False
            )
            
            steer_observations_df = pd.concat([steer_observations_df, y_samples, n_samples])
        
        # Get remaining observations for test set
        test_observations_df = self.observations_df[~self.observations_df['observation_id'].isin(steer_observations_df['observation_id'])]
        
        kwargs = {
            'max_personas': self.max_personas,
        }
        return (
            StatementsDataset(**kwargs, personas_df=self.personas_df, observations_df=steer_observations_df),
            StatementsDataset(**kwargs, personas_df=self.personas_df, observations_df=test_observations_df)
        )
