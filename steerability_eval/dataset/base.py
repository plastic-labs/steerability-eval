SystemResponse = str
PersonaId = str
ScenarioId = str
ObservationId = str

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
