from typing import List, Union
import time
from time import sleep
import asyncio

from steerability_eval.steerable.base import BaseSteeredSystem, BaseSteerableSystem
from steerability_eval.dataset.base import Persona, Observation, SystemResponse
from steerability_eval.dataset.persona_framework import generate_short_hash
from honcho import Honcho, AsyncHoncho
from honcho.types import App

AGREE_STR: str = 'Y'
DISAGREE_STR: str = 'N'

HONCHO_ENVIRONMENT: str = 'demo'
HONCHO_APP_ID: str = 'steerability-eval-2'

class HonchoSteerable(BaseSteerableSystem):
    def __init__(
        self,
        wait_on_init: bool = False,
        honcho_environment: str = HONCHO_ENVIRONMENT,
        honcho_app_id: str = HONCHO_APP_ID,
        verbose: bool = False
    ):
        self.wait_on_init: bool = wait_on_init
        self.honcho = Honcho(environment=honcho_environment) # type: ignore
        self.app = self.honcho.apps.get_or_create(name=honcho_app_id)
        self.verbose: bool = verbose

    def steer(self, persona: Persona, observations: List[Observation]) -> 'HonchoSteeredSystem':
        return HonchoSteeredSystem(
            persona,
            self,
            observations,
            self.honcho,
            self.app,
            self.verbose,
            self.wait_on_init
        )

class HonchoSteeredSystem(BaseSteeredSystem):
    def __init__(
        self,
        persona: Persona,
        steerable_system: HonchoSteerable,
        observations: List[Observation],
        honcho: Honcho,
        app: App,
        verbose: bool = False,
        wait_on_init: bool = False
    ):
        super().__init__(persona, steerable_system, observations)
        self.verbose: bool = verbose
        if verbose:
            print(f'Steering to {persona.persona_description}')
        self.honcho = honcho
        self.app = app
        self.username: str = generate_short_hash(f'{persona.persona_description}-{time.time()}')
        self.user = self.honcho.apps.users.get_or_create(app_id=self.app.id, name=self.username)
        self.session = self.honcho.apps.users.sessions.create(app_id=self.app.id, user_id=self.user.id)
        self.send_steering_messages(observations)
        if wait_on_init:
            self.user_representation: str = self.wait_for_user_representation()

    def send_steering_messages(self, observations: List[Observation]) -> None:
        for observation in observations:
            ai_message = f'Do you agree with this statement? "{observation.response}". Respond with "Y" or "N" and nothing else.'
            message = self.honcho.apps.users.sessions.messages.create(
                session_id=self.session.id,
                app_id=self.app.id,
                user_id=self.user.id,
                content=ai_message,
                is_user=False
            )
            user_message = f'{observation.correct_response}'
            message = self.honcho.apps.users.sessions.messages.create(
                session_id=self.session.id,
                app_id=self.app.id,
                user_id=self.user.id,
                content=user_message,
                is_user=True
            )
        self.final_steering_message_id = message.id

    def run_inference(self, observation: Observation) -> SystemResponse:
        query = f'''
        If you had to make your best guess based on your knowledge of the user, would they agree with the statement: "{observation.response}"?
        Respond with "Y" or "N" and nothing else.
        '''
        response = self.honcho.apps.users.sessions.chat(
            session_id=self.session.id,
            app_id=self.app.id,
            user_id=self.user.id,
            queries=[query]
        )
        return SystemResponse(response.content)

    def wait_for_user_representation(self) -> str:
        have_representation = False
        while not have_representation:
            metamessages = self.honcho.apps.users.sessions.metamessages.list(
                session_id=self.session.id,
                app_id=self.app.id,
                metamessage_type='user_representation',
                user_id=self.user.id
            )
            if metamessages.items:
                final_metamessage = metamessages.items[-1]
                if final_metamessage.message_id == self.final_steering_message_id:
                    have_representation = True
                    return metamessages.items[-1].content
            sleep(0.5)
        return ''

    def wait_until_ready(self) -> str:
        return self.wait_for_user_representation()


class AsyncHonchoSteerable(BaseSteerableSystem):
    def __init__(
        self,
        wait_on_init: bool = True,
        honcho_environment: str = HONCHO_ENVIRONMENT,
        honcho_app_id: str = HONCHO_APP_ID,
        verbose: bool = False
    ):
        self.wait_on_init: bool = wait_on_init
        self.honcho = AsyncHoncho(environment=honcho_environment) # type: ignore
        self.app_id: str = honcho_app_id
        self.verbose: bool = verbose


    async def steer(self, persona: Persona, observations: List[Observation]) -> 'AsyncHonchoSteeredSystem':
        steered_system = await AsyncHonchoSteeredSystem.create(
            persona,
            self,
            observations,
            self.honcho,
            self.verbose,
            self.wait_on_init
        )
        return steered_system

class AsyncHonchoSteeredSystem(BaseSteeredSystem):
    def __init__(
        self,
        persona: Persona,
        steerable_system: AsyncHonchoSteerable,
        observations: List[Observation],
        honcho: AsyncHoncho,
        verbose: bool = False,
        user=None,
        session=None
    ):
        super().__init__(persona, steerable_system, observations)
        self.verbose: bool = verbose
        if verbose:
            print(f'Steering to {persona.persona_description}')
        self.honcho = honcho  # type: ignore
        self.username: str = generate_short_hash(f'{persona.persona_description}-{time.time()}')
        self.user = user  # type: ignore
        self.session = session  # type: ignore
        self.app = None  # type: ignore
        self.user_representation: str = ''

    @classmethod
    async def create(
        cls,
        persona: Persona,
        steerable_system: AsyncHonchoSteerable,
        observations: List[Observation],
        honcho: AsyncHoncho,
        verbose: bool = False,
        wait_on_init: bool = False
    ) -> 'AsyncHonchoSteeredSystem':
        instance = cls(persona, steerable_system, observations, honcho, verbose, wait_on_init)
        instance.app = await instance.honcho.apps.get_or_create(name=HONCHO_APP_ID)
        instance.user = await instance.honcho.apps.users.get_or_create(app_id=instance.app.id, name=instance.username)
        instance.session = await instance.honcho.apps.users.sessions.create(
            app_id=instance.app.id,
            user_id=instance.user.id
        )
        await instance.send_steering_messages(observations)
        if wait_on_init:
            instance.user_representation = await instance.wait_for_user_representation()
        return instance

    async def send_steering_messages(self, observations: List[Observation]) -> None:
        for observation in observations:
            ai_message = f'Do you agree with this statement? "{observation.response}". Respond with "Y" or "N" and nothing else.'
            print(ai_message)
            message = await self.honcho.apps.users.sessions.messages.create(
                session_id=self.session.id,
                app_id=self.app.id,
                user_id=self.user.id,
                content=ai_message,
                is_user=False
            )
            user_message = f'{observation.correct_response}'
            print(user_message)
            message = await self.honcho.apps.users.sessions.messages.create(
                session_id=self.session.id,
                app_id=self.app.id,
                user_id=self.user.id,
                content=user_message,
                is_user=True
            )
        self.final_steering_message_id = message.id

    async def run_inference(self, observation: Observation) -> SystemResponse:
        query = f'''
        If you had to make your best guess based on your knowledge of the user, would they agree with the statement: "{observation.response}"?
        Respond with "Y" or "N" and nothing else.
        '''
        response = await self.honcho.apps.users.sessions.chat(
            session_id=self.session.id,
            app_id=self.app.id,
            user_id=self.user.id,
            queries=[query]
        )
        return SystemResponse(response.content)

    async def wait_for_user_representation(self) -> str:
        have_representation = False
        while not have_representation:
            metamessages = await self.honcho.apps.users.sessions.metamessages.list(
                session_id=self.session.id,
                app_id=self.app.id,
                metamessage_type='user_representation',
                user_id=self.user.id
            )
            print(metamessages)
            if metamessages.items:
                final_metamessage = metamessages.items[-1]
                if final_metamessage.message_id == self.final_steering_message_id:
                    have_representation = True
                    return metamessages.items[-1].content
            sleep(0.5)
        return ''

    async def wait_until_ready(self) -> str:
        return await self.wait_for_user_representation()
