from typing import List, Union, Dict, Any, Type, Optional
import time
from time import sleep
import asyncio

from steerability_eval.steerable.base import BaseSteeredSystem, BaseSteerableSystem
from steerability_eval.dataset.base import Persona, Observation, SystemResponse
from steerability_eval.dataset.persona_framework import generate_short_hash
from honcho import Honcho, AsyncHoncho
from honcho.types import App
from steerability_eval.steerable.state import SteeredSystemState

AGREE_STR: str = 'Y'
DISAGREE_STR: str = 'N'

HONCHO_ENVIRONMENT: str = 'demo'
HONCHO_APP_ID: str = 'steerability-eval-2'

class HonchoState(SteeredSystemState):
    """State for a Honcho steered system"""
    def __init__(self, 
                 persona: Persona,
                 environment: str, 
                 app_id: str,
                 username: str,
                 user_representation: str,
                 observations: List[Observation]):
        self.persona = persona
        self.observations = observations
        self.environment = environment
        self.app_id = app_id
        self.username = username
        self.user_representation = user_representation

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary"""
        return {
            'persona': self.persona.to_dict(),
            'environment': self.environment,
            'app_id': self.app_id,
            'username': self.username,
            'user_representation': self.user_representation,
            'observations': [observation.to_dict() for observation in self.observations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HonchoState':
        """Create state from dictionary"""
        persona = Persona(**data['persona'])
        observations = [Observation.from_dict(observation_data) for observation_data in data['observations']]
        return cls(
            persona=persona,
            observations=observations,
            environment=data['environment'],
            app_id=data['app_id'],
            username=data['username'],
            user_representation=data['user_representation']
        )

class HonchoSteerable(BaseSteerableSystem):
    def __init__(
        self,
        wait_on_init: bool = True,
        honcho_environment: str = HONCHO_ENVIRONMENT,
        honcho_app_id: str = HONCHO_APP_ID,
        verbose: bool = False
    ):
        self.wait_on_init: bool = wait_on_init
        self.honcho = Honcho(environment=honcho_environment) # type: ignore
        self.environment = honcho_environment
        self.app = self.honcho.apps.get_or_create(name=honcho_app_id)
        self.verbose: bool = verbose

    @staticmethod
    def supports_async_steering() -> bool:
        """Whether this system supports async steering"""
        return False

    @staticmethod
    def supports_batch_inference() -> bool:
        """Whether this system supports batch inference"""
        return True

    @staticmethod
    def supports_async_inference() -> bool:
        """Whether this system supports async inference"""
        return False

    @staticmethod
    def supports_saving_state() -> bool:
        """Whether this system supports saving state"""
        return True
        
    def steer(self, persona: Persona, observations: List[Observation]) -> 'HonchoSteeredSystem':
        steered_system = HonchoSteeredSystem(
            persona,
            self,
            observations,
            self.honcho,
            self.app,
            self.environment,
            self.verbose,
            self.wait_on_init
        )
        steered_system.send_steering_messages(observations)
        if self.wait_on_init:
            steered_system.user_representation = steered_system.wait_for_user_representation()
        return steered_system

    def get_steered_state_class(self) -> Type[SteeredSystemState]:
        """Get the state class for this steerable system"""
        return HonchoState

    def create_steered_from_state(self, state: HonchoState) -> 'HonchoSteeredSystem':
        """Create a steered system from saved state"""
        steered_system = HonchoSteeredSystem(
            persona=state.persona,
            steerable_system=self,
            observations=state.observations,
            honcho=self.honcho,
            app=self.app,
            environment=self.environment,
            verbose=self.verbose,
            wait_on_init=self.wait_on_init,
            username=state.username  # Pass the saved username
        )
        steered_system.user = self.honcho.apps.users.get_or_create(app_id=self.app.id, name=state.username)
        steered_system.session = self.honcho.apps.users.sessions.create(app_id=self.app.id, user_id=steered_system.user.id)
        steered_system.user_representation = state.user_representation
        return steered_system

class HonchoSteeredSystem(BaseSteeredSystem):
    def __init__(
        self,
        persona: Persona,
        steerable_system: HonchoSteerable,
        observations: List[Observation],
        honcho: Honcho,
        app: App,
        environment: str,
        verbose: bool = False,
        wait_on_init: bool = False,
        username: Optional[str] = None
    ):
        super().__init__(persona, steerable_system, observations)
        self.environment = environment
        self.verbose = verbose
        if verbose:
            print(f'Steering to {persona.persona_description}')
        self.honcho = honcho
        self.app = app
        self.observations = observations
        
        self.username = username or generate_short_hash(f'{persona.persona_description}-{time.time()}')
        
        self.user = self.honcho.apps.users.get_or_create(app_id=self.app.id, name=self.username)
        self.session = self.honcho.apps.users.sessions.create(app_id=self.app.id, user_id=self.user.id)
        self.wait_on_init = wait_on_init
        self.user_representation: str = ''

    def send_steering_messages(self, observations: List[Observation]) -> None:
        if self.verbose:
            print(f'Sending steering messages')
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

    def get_state(self) -> HonchoState:
        """Get current state of the steered system"""
        return HonchoState(
            persona=self.persona,
            observations=self.observations,
            environment=self.environment,
            app_id=self.app.id,
            username=self.username,
            user_representation=self.user_representation
        )

class AsyncHonchoSteerable(BaseSteerableSystem):
    def __init__(
        self,
        wait_on_init: bool = True,
        environment: str = HONCHO_ENVIRONMENT,
        app_id: str = HONCHO_APP_ID,
        verbose: bool = False
    ):
        self.wait_on_init: bool = wait_on_init
        self.honcho = AsyncHoncho(environment=environment) # type: ignore
        self.environment = environment
        self.app_id: str = app_id
        self.verbose: bool = verbose

    @staticmethod
    def supports_async_inference() -> bool:
        """Whether this system supports async inference"""
        return True
    
    @staticmethod
    def supports_batch_inference() -> bool:
        """Whether this system supports batch inference"""
        return True

    @staticmethod
    def supports_async_steering() -> bool:
        """Whether this system supports async steering"""
        return True

    @staticmethod
    def get_steered_state_class() -> Type[SteeredSystemState]:
        """Get the state class for this steerable system"""
        return HonchoState

    @staticmethod
    def supports_saving_state() -> bool:
        """Whether this system supports saving state"""
        return True

    async def create_steered_from_state_async(self, state: HonchoState) -> 'AsyncHonchoSteeredSystem':
        """Create a steered system from saved state"""
        steered_system = AsyncHonchoSteeredSystem(
            persona=state.persona,
            steerable_system=self,
            observations=state.observations,
            honcho=self.honcho,
            environment=self.environment,
            verbose=self.verbose,
            username=state.username,
        )
        steered_system.app = await steered_system.honcho.apps.get_or_create(name=self.app_id)
        steered_system.user = await steered_system.honcho.apps.users.get_or_create(app_id=steered_system.app.id, name=steered_system.username)
        steered_system.session = await steered_system.honcho.apps.users.sessions.create(app_id=steered_system.app.id, user_id=steered_system.user.id)
        steered_system.user_representation = state.user_representation
        return steered_system

    async def steer_async(self, persona: Persona, observations: List[Observation]) -> 'AsyncHonchoSteeredSystem':
        steered_system = await AsyncHonchoSteeredSystem.create(
            persona=persona,
            steerable_system=self,
            observations=observations,
            honcho=self.honcho,
            environment=self.environment,
            verbose=self.verbose,
            wait_on_init=self.wait_on_init
        )
        return steered_system



class AsyncHonchoSteeredSystem(BaseSteeredSystem):
    def __init__(
        self,
        persona: Persona,
        steerable_system: AsyncHonchoSteerable,
        observations: List[Observation],
        honcho: AsyncHoncho,
        environment: str,
        verbose: bool = False,
        username: Optional[str] = None,
    ):
        super().__init__(persona, steerable_system, observations)
        self.environment = environment
        self.verbose: bool = verbose
        self.observations = observations
        self.honcho = honcho
        self.username = username or generate_short_hash(f'{persona.persona_description}-{time.time()}')
        self.app: Optional[App] = None
        self.user: Optional[User] = None # type: ignore
        self.session: Optional[Session] = None # type: ignore
        self.user_representation: str = ''

    def get_state(self) -> HonchoState:
        """Get current state of the steered system"""
        return HonchoState(
            persona=self.persona,
            observations=self.observations,
            environment=self.environment,
            app_id=self.app.id if self.app else '',
            username=self.username,
            user_representation=self.user_representation
        )


    @classmethod
    async def create(
        cls,
        persona: Persona,
        steerable_system: AsyncHonchoSteerable,
        observations: List[Observation],
        environment: str,
        honcho: AsyncHoncho,
        verbose: bool = False,
        wait_on_init: bool = False,
        username: Optional[str] = None
    ) -> 'AsyncHonchoSteeredSystem':
        instance = cls(
            persona=persona, 
            steerable_system=steerable_system, 
            observations=observations, 
            honcho=honcho, 
            environment=environment,
            verbose=verbose,
            username=username
        )
        instance.app = await instance.honcho.apps.get_or_create(name=steerable_system.app_id)
        instance.user = await instance.honcho.apps.users.get_or_create(
            app_id=instance.app.id, 
            name=instance.username
        )
        print(f'Creating steered system for {instance.username}')
        instance.session = await instance.honcho.apps.users.sessions.create(
            app_id=instance.app.id,
            user_id=instance.user.id
        )
        await instance.send_steering_messages(observations)
        if wait_on_init:
            instance.user_representation = await instance.wait_for_user_representation()
        return instance

    def supports_batch_inference(self) -> bool:
        return True

    async def send_steering_messages(self, observations: List[Observation]) -> None:
        for observation in observations:
            ai_message = f'Do you agree with this statement? "{observation.response}". Respond with "Y" or "N" and nothing else.'
            ai_message = ai_message.replace('sensuality', 'appeal')
            ai_message = ai_message.replace('exploit vulnerabilities', 'tackle opportunities')
            message = await self.honcho.apps.users.sessions.messages.create(
                session_id=self.session.id,
                app_id=self.app.id,
                user_id=self.user.id,
                content=ai_message,
                is_user=False
            )
            user_message = f'{observation.correct_response}'
            message = await self.honcho.apps.users.sessions.messages.create(
                session_id=self.session.id,
                app_id=self.app.id,
                user_id=self.user.id,
                content=user_message,
                is_user=True
            )
        self.final_steering_message_id = message.id

    async def run_inference_async(self, observation: Observation) -> SystemResponse:
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

    async def run_batch_inference_async(self, observations: List[Observation]) -> List[SystemResponse]:
        queries = [f'''
        If you had to make your best guess based on your knowledge of the user, would they agree with the statement: "{observation.response}"?
        Respond with "Y" or "N" and nothing else. Separate each response with a single newline.
        ''' for observation in observations]
        response = await self.honcho.apps.users.sessions.chat(
            session_id=self.session.id,
            app_id=self.app.id,
            user_id=self.user.id,
            queries=queries
        )
        if '\n\n' in response.content:
            return [SystemResponse(individual_response) for individual_response in response.content.split('\n\n')]
        elif '\n' in response.content:
            return [SystemResponse(individual_response) for individual_response in response.content.split('\n')]
        else:
            raise ValueError(f'Unexpected response format: {response.content}')

    async def wait_for_user_representation(self) -> str:
        have_representation = False
        while not have_representation:
            metamessages = await self.honcho.apps.users.sessions.metamessages.list(
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

    async def wait_until_ready(self) -> str:
        return await self.wait_for_user_representation()
