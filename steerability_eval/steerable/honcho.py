from typing import List
import time
from time import sleep

from steerability_eval.steerable.base import BaseSteeredSystem, BaseSteerableSystem
from steerability_eval.dataset.base import Persona, Observation, SystemResponse
from steerability_eval.dataset.persona_framework import generate_short_hash
from honcho import Honcho
from honcho.types import App

AGREE_STR: str = 'Y'
DISAGREE_STR: str = 'N'

HONCHO_ENVIRONMENT: str = 'demo'
HONCHO_APP_ID: str = 'steerability-eval'

class HonchoSteerable(BaseSteerableSystem):
    def __init__(self, honcho_environment: str = HONCHO_ENVIRONMENT, honcho_app_id: str = HONCHO_APP_ID, verbose: bool = False):
        self.honcho = Honcho(environment=honcho_environment) # type: ignore
        self.app = self.honcho.apps.get_or_create(name=honcho_app_id)
        self.verbose = verbose

    def steer(self, persona: Persona, observations: List[Observation], wait_on_init: bool = False) -> 'HonchoSteeredSystem':
        return HonchoSteeredSystem(persona, self, observations, self.honcho, self.app, self.verbose, wait_on_init)


class HonchoSteeredSystem(BaseSteeredSystem):
    def __init__(self,
                 persona: Persona,
                 steerable_system: HonchoSteerable,
                 observations: List[Observation],
                 honcho: Honcho,
                 app: App,
                 verbose: bool = False,
                 wait_on_init: bool = False):
        super().__init__(persona, steerable_system, observations)
        self.verbose = verbose
        if verbose:
            print(f'Steering to {persona.persona_description}')
        self.honcho = honcho
        self.app = app
        self.username = generate_short_hash(f'{persona.persona_description}-{time.time()}')
        self.user = self.honcho.apps.users.get_or_create(app_id=self.app.id, name=self.username)
        self.session = self.honcho.apps.users.sessions.create(app_id=self.app.id, user_id=self.user.id)
        self.send_steering_messages(observations)
        if wait_on_init:
            self.user_representation = self.wait_for_user_representation()

    def send_steering_messages(self, observations: List[Observation]) -> None:
        for observation in observations:
            ai_message = f'Do you agree with this statement? "{observation.response}". Respond with "Y" or "N" and nothing else.'
            self.honcho.apps.users.sessions.messages.create(session_id=self.session.id,
                                                            app_id=self.app.id,
                                                            user_id=self.user.id,
                                                            content=ai_message,
                                                            is_user=False)
            user_message = f'{observation.correct_response}'
            self.honcho.apps.users.sessions.messages.create(session_id=self.session.id,
                                                            app_id=self.app.id,
                                                            user_id=self.user.id,
                                                            content=user_message,
                                                            is_user=True)

    def run_inference(self, observation: Observation) -> SystemResponse:
        query = f'''
        If you had to make your best guess based on your knowledge of the user, would they agree with the statement: "{observation.response}"?
        Respond with "Y" or "N" and nothing else.
        '''
        response = self.honcho.apps.users.sessions.chat(session_id=self.session.id,
                                                       app_id=self.app.id,
                                                       user_id=self.user.id,
                                                       queries=[query])
        return SystemResponse(response.content)

    def wait_for_user_representation(self) -> str:
        have_representation = False
        while not have_representation:
            metamessages = self.honcho.apps.users.sessions.metamessages.list(session_id=self.session.id, app_id=self.app.id, metamessage_type='user_representation', user_id=self.user.id)
            if len(metamessages.items) > 0:
                have_representation = True
                return metamessages.items[-1].content
            else:
                sleep(0.5)
        return ''

    def wait_until_ready(self) -> None:
        ready = False
        while not ready:
            self.user_representation = self.wait_for_user_representation()
            if self.user_representation:
                ready = True
