from typing import List, Dict, Set, Tuple

import pandas as pd

from steerability_eval.dataset.base import BaseDataset, Persona, Observation, PersonaId, ObservationId, ScenarioId
from steerability_eval.util import generate_short_hash

MAX_PERSONAS = 10
MAX_CONTEXTS_PER_THEME = 10
MAX_SCENARIOS_PER_CONTEXT = 3


class W5TFDataset(BaseDataset):
    def __init__(
        self,
        personas_df: pd.DataFrame,
        observations_df: pd.DataFrame,
        max_personas: int = MAX_PERSONAS,
        random_state: int = 42
    ):
        self.max_personas = max_personas
        self.personas_df = personas_df
        self.observations_df = observations_df
        self.random_state = random_state

    @classmethod
    def from_csv(cls,
                 personas_path: str,
                 observations_path: str,
                 max_personas: int = MAX_PERSONAS,
                 max_contexts_per_theme: int = MAX_CONTEXTS_PER_THEME,
                 max_scenarios_per_context: int = MAX_SCENARIOS_PER_CONTEXT,
                 random_state: int = 42):
        personas_df = cls.load_personas(personas_path, max_personas, random_state)
        persona_ids = list(personas_df['persona_id'].unique())
        observations_df = cls.load_observations(
            observations_path,
            persona_ids,
            max_contexts_per_theme,
            max_scenarios_per_context,
            random_state
        )
        return cls(personas_df,
                   observations_df,
                   max_personas,
                   random_state)

    @classmethod
    def load_personas(cls, personas_path: str, max_personas: int = MAX_PERSONAS, random_state: int = 42) -> pd.DataFrame:
        with open(personas_path, 'r') as f:
            df = pd.read_csv(f)
        n_personas = min(max_personas, len(df))
        df = df.sample(n=n_personas, random_state=random_state, replace=False)
        df.rename(columns={'framework_name': 'framework'}, inplace=True)
        return df

    @classmethod
    def load_observations(cls,
                          observations_path: str,
                          persona_ids: List[PersonaId],
                          max_contexts_per_theme: int,
                          max_scenarios_per_context: int,
                          random_state: int = 42) -> pd.DataFrame:
        with open(observations_path, 'r') as f:
            df = pd.read_csv(f)
        df = df[df['persona_id'].isin(persona_ids)]
        
        selected_contexts = df.groupby('theme_id')['context_id'].apply(
            lambda x: x.drop_duplicates().sample(
                n=min(max_contexts_per_theme, len(x.unique())),
                replace=False,
                random_state=random_state
            )
        ).reset_index(level=1, drop=True).explode()
            
        # Filter the dataframe to keep only the selected contexts
        df = df[df['context_id'].isin(selected_contexts)]
        
        # Group by context and randomly select scenarios
        selected_scenarios = df.groupby('context_id')['scenario_id'].apply(
            lambda x: x.drop_duplicates().sample(
                n=min(max_scenarios_per_context, len(x.unique())),
                replace=False,
                random_state=random_state
            )
        ).reset_index(level=1, drop=True).explode()
        
        # Keep only the selected scenarios
        df = df[df['scenario_id'].isin(selected_scenarios)]

        df['scenario'] = df['context'] + df['scenario']

        # Combine context and scenario fields and IDs
        df['observation_id'] = df.apply(
            lambda row: generate_short_hash( f"{row['persona_id']}_{row['scenario_id']}_{row['correct_response']}"),
            axis=1
        )

        return df

    @property
    def personas(self) -> List[Persona]:
        return [Persona.from_row(row) for _, row in self.personas_df.iterrows()]

    def get_persona(self, persona_id: PersonaId) -> Persona:
        return Persona.from_row(self.personas_df[self.personas_df['persona_id'] == persona_id].iloc[0])

    def get_observations_by_persona(self, persona: Persona) -> List[Observation]:
        return [Observation.from_row(row) 
                for _, row in 
                self.observations_df[ self.observations_df['persona_id'] == persona.persona_id].iterrows()
                ]

    def split(self, n_steer_observations_per_persona: int = 10, random_state: int = 42) -> Tuple['W5TFDataset', 'W5TFDataset']:
        steer_observations_df = self.observations_df.groupby('persona_id').sample(n=n_steer_observations_per_persona, random_state=random_state)
        test_observations_df = self.observations_df[~self.observations_df['observation_id'].isin(steer_observations_df['observation_id'])]
        kwargs = {
            'max_personas': self.max_personas,
        }
        return (
            W5TFDataset(**kwargs, personas_df=self.personas_df, observations_df=steer_observations_df),
            W5TFDataset(**kwargs, personas_df=self.personas_df, observations_df=test_observations_df)
        )
