from typing import List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from steerability_eval.steerable.base import BaseSteerableSystem, BaseSteeredSystem
from steerability_eval.dataset import Persona, Observation
from steerability_eval.util.llm import get_chat_model
from steerability_eval.eval import AGREE_STR, DISAGREE_STR


DEFAULT_PROVIDER = 'google'


class FewShotSteerable(BaseSteerableSystem):
    def __init__(
        self,
        llm_provider: str = DEFAULT_PROVIDER,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = 0.0
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature

    def steer(self, persona: Persona, steer_observations: List[Observation]) -> BaseSteeredSystem:
        return FewShotSteeredSystem(persona, self, steer_observations)


class FewShotSteeredSystem(BaseSteeredSystem):
    def __init__(
        self,
        persona: Persona,
        steerable_system: FewShotSteerable,
        steer_observations: List[Observation],
    ):
        super().__init__(persona, steerable_system, steer_observations)
        self.llm_provider = steerable_system.llm_provider
        self.model = steerable_system.model
        self.api_key = steerable_system.api_key
        self.base_url = steerable_system.base_url
        self.temperature = steerable_system.temperature
        self.prompt = self.generate_prompt(persona, steer_observations)
        self.llm = get_chat_model(self.llm_provider, self.model, self.api_key, self.base_url)
        self.llm_chain = self.prompt | self.llm | JsonOutputParser()

    def generate_prompt(self, persona: Persona, steer_observations: List[Observation]) -> PromptTemplate:
        scenario_string = '\n'.join(
            [f'{i+1}. Scenario: {o.scenario}\nResponse: {o.response}'
             for i, o in enumerate(steer_observations)])
        prompt_str = f"""
        You are role playing as a persona described as follows:
        {persona.persona_description}

        The following are factual observations of how this persona responded to a series of scenarios:
        {scenario_string}

        You will now be given a new scenario, and a hypothetical response.
        Your job is to determine, based on your understanding of the persona, whether the response is consistent with the persona.
        """

        prompt_str += """
        Respond in valid JSON format with the following keys:
        - "agree": true if the response is consistent with the persona, false otherwise
        Respond in valid JSON and nothing else.

        Scenario: {scenario}
        Response: {response}
        """
        return PromptTemplate(template=prompt_str, input_variables=[])

    def run_inference(self, observation: Observation) -> str:
        llm_response = self.llm_chain.invoke(
            {"scenario": observation.scenario, "response": observation.response}
        )['agree']
        return AGREE_STR if llm_response else DISAGREE_STR

    async def run_inference_async(self, observation: Observation) -> str:
        llm_response = await self.llm_chain.ainvoke(
            {"scenario": observation.scenario, "response": observation.response}
        )
        response = llm_response['agree']
        return AGREE_STR if response else DISAGREE_STR
