from typing import List, Optional, Dict, Any, Type

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from steerability_eval.steerable.base import BaseSteerableSystem, BaseSteeredSystem
from steerability_eval.steerable.state import SteeredSystemState
from steerability_eval.dataset.base import Persona, Observation, SystemResponse
from steerability_eval.util.llm import get_chat_model

DEFAULT_PROVIDER: str = 'google'
AGREE_STR: str = 'Y'
DISAGREE_STR: str = 'N'

class FewShotState(SteeredSystemState):
    """State for a FewShot steered system"""
    def __init__(self, 
                 persona: Persona,
                 observations: List[Observation],
                 include_persona: bool,
                 include_observations: bool):
        self.persona = persona
        self.observations = observations
        self.include_persona = include_persona
        self.include_observations = include_observations

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary"""
        return {
            'persona': self.persona.to_dict(),
            'observations': [obs.to_dict() for obs in self.observations],
            'include_persona': self.include_persona,
            'include_observations': self.include_observations
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FewShotState':
        """Create state from dictionary"""
        return cls(
            persona=Persona(**data['persona']),
            observations=[Observation.from_dict(obs) for obs in data['observations']],
            include_persona=data['include_persona'],
            include_observations=data['include_observations']
        )

class FewShotSteerable(BaseSteerableSystem):
    def __init__(self,
                 llm_provider: str = 'google',
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.0,
                 include_persona: bool = True,
                 include_observations: bool = True,
                 verbose: bool = False):
        self.llm_provider = llm_provider
        self.include_persona = include_persona
        self.include_observations = include_observations
        self.verbose = verbose
        self.llm = get_chat_model(provider=llm_provider,
                                  model=model,
                                  api_key=api_key,
                                  base_url=base_url,
                                  temperature=temperature)

    @staticmethod
    def supports_async_steering() -> bool:
        return False

    @staticmethod
    def supports_batch_inference() -> bool:
        return False

    @staticmethod
    def supports_async_inference() -> bool:
        return True

    @staticmethod
    def supports_saving_state() -> bool:
        return True

    @staticmethod
    def get_steered_state_class() -> Type[SteeredSystemState]:
        return FewShotState

    def create_steered_from_state(self, state: FewShotState) -> 'FewShotSteeredSystem':
        return FewShotSteeredSystem(
            persona=state.persona,
            steerable_system=self,
            observations=state.observations,
            llm=self.llm,
            include_persona=state.include_persona,
            include_observations=state.include_observations
        )

    def steer(self, persona: Persona, observations: List[Observation]) -> 'FewShotSteeredSystem':
        return FewShotSteeredSystem(
            persona=persona,
            steerable_system=self,
            observations=observations,
            llm=self.llm,
            include_persona=self.include_persona,
            include_observations=self.include_observations
        )

class FewShotSteeredSystem(BaseSteeredSystem):
    def __init__(self,
                 persona: Persona,
                 steerable_system: FewShotSteerable,
                 observations: List[Observation],
                 llm: Any,
                 include_persona: bool,
                 include_observations: bool):
        super().__init__(persona, steerable_system, observations)
        self.llm = llm
        self.observations = observations
        self.persona = persona
        self.include_persona = include_persona
        self.include_observations = include_observations
        self.prompt = build_prompt(self, observations)
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def get_state(self) -> FewShotState:
        return FewShotState(
            persona=self.persona,
            observations=self.observations,
            include_persona=self.include_persona,
            include_observations=self.include_observations
        )

    def run_inference(self, observation: Observation) -> SystemResponse:
        response = self.chain.invoke({"response": observation.response})
        return SystemResponse(AGREE_STR if response["agree"] else DISAGREE_STR)

    async def run_inference_async(self, observation: Observation) -> str:
        llm_response = await self.chain.ainvoke({"response": observation.response})
        return AGREE_STR if llm_response["agree"] else DISAGREE_STR

    def wait_until_ready(self) -> None:
        pass

def build_prompt(steered_system: FewShotSteeredSystem,
                 observations: List[Observation]) -> PromptTemplate:
    prompt_parts: List[str] = []
        
    if steered_system.include_persona:
        prompt_parts.append(f"""
        You are role playing as a persona described as follows:
        {steered_system.persona.persona_description}
        """)
    else:
        prompt_parts.append("You are role playing as a persona.")

    if steered_system.include_observations:
        agree_string = '\n'.join(
            [o.response for o in observations if o.correct_response == AGREE_STR]
        )
        disagree_string = '\n'.join(
            [o.response for o in observations if o.correct_response == DISAGREE_STR]
        )
            
        prompt_parts.extend([
            f"""
            The following are statements that this persona has made and many other people would disagree with:
            {agree_string}
            """,
            f"""
            The following are statements that other people would agree with, but this persona would disagree with:
            {disagree_string}
            """
        ])

    prompt_parts.append("""
        You will now be given a new statement, and a hypothetical response.
        Your job is to determine, based on your understanding of the persona, whether the response is consistent with the persona.
        
        Respond in valid JSON format with the following keys:
        - "agree": true if the response is consistent with the persona, false otherwise
        Respond in valid JSON and nothing else.

        Response: {response}
        """)

    return PromptTemplate(template="\n".join(prompt_parts), input_variables=["response"])