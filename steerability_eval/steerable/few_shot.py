from typing import List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from steerability_eval.steerable.base import BaseSteerableSystem, BaseSteeredSystem
from steerability_eval.dataset.base import Persona, Observation
from steerability_eval.util.llm import get_chat_model

DEFAULT_PROVIDER: str = 'google'
AGREE_STR: str = 'Y'
DISAGREE_STR: str = 'N'

class FewShotSteerable(BaseSteerableSystem):
    def __init__(
        self,
        llm_provider: str = DEFAULT_PROVIDER,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        include_persona: bool = True,
        include_observations: bool = True
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.include_persona = include_persona
        self.include_observations = include_observations

    def steer(self, persona: Persona, observations: List[Observation]) -> BaseSteeredSystem:
        return FewShotSteeredSystem(
            persona=persona,
            steerable_system=self,
            observations=observations if self.include_observations else []
        )

class FewShotSteeredSystem(BaseSteeredSystem):
    def __init__(
        self,
        persona: Persona,
        steerable_system: FewShotSteerable,
        observations: List[Observation]
    ):
        super().__init__(persona, steerable_system, observations)
        self.llm_provider = steerable_system.llm_provider
        self.model = steerable_system.model
        self.api_key = steerable_system.api_key
        self.base_url = steerable_system.base_url
        self.temperature = steerable_system.temperature
        self.include_persona = steerable_system.include_persona
        self.include_observations = steerable_system.include_observations
        self.prompt = self.generate_prompt(persona, observations)
        self.llm = get_chat_model(self.llm_provider, self.model, self.api_key, self.base_url)
        self.llm_chain = self.prompt | self.llm | JsonOutputParser()

    def generate_prompt(self, persona: Persona, observations: List[Observation]) -> PromptTemplate:
        prompt_parts: List[str] = []
        
        if self.include_persona:
            prompt_parts.append(f"""
            You are role playing as a persona described as follows:
            {persona.persona_description}
            """)
        else:
            prompt_parts.append("You are role playing as a persona:")

        if self.include_observations:
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

    def run_inference(self, observation: Observation) -> str:
        llm_response = self.llm_chain.invoke({"response": observation.response})
        return AGREE_STR if llm_response["agree"] else DISAGREE_STR

    async def run_inference_async(self, observation: Observation) -> str:
        llm_response = await self.llm_chain.ainvoke({"response": observation.response})
        return AGREE_STR if llm_response["agree"] else DISAGREE_STR

    def wait_until_ready(self) -> None:
        pass
