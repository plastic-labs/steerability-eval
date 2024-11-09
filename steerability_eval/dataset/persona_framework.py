import hashlib

def generate_short_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]

class BasePersonaFramework:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.framework_id = generate_short_hash(name)

    def get_personas(self):
        raise NotImplementedError('This method must be implemented in the subclass.')


class Persona():
    def __init__(self, framework: BasePersonaFramework, persona_description: str):
        self.framework = framework
        self.persona_id = generate_short_hash(f'{framework.name} {persona_description}')
        self.framework_id = framework.framework_id
        self.persona_description = persona_description
    

class MBTI(BasePersonaFramework):
    framework_name = 'MBTI'
    framework_description = '''
    The Myers-Briggs Type Indicator (MBTI) is a personality assessment system that categorizes individuals into 16 distinct personality types based on four dichotomies: Extraversion (E) vs. Introversion (I), Sensing (S) vs. Intuition (N), Thinking (T) vs. Feeling (F), and Judging (J) vs. Perceiving (P). Each type is represented by a four-letter code (e.g., INTJ, ESFP) that indicates a person's preferences in how they perceive the world, process information, make decisions, and structure their lives. These types are believed to influence communication styles, problem-solving approaches, and interpersonal dynamics, providing a framework for understanding individual differences in behavior and cognition.
    '''
    def __init__(self):
        super().__init__(self.framework_name, self.framework_description)
        letter_1 = ['I', 'E']
        letter_2 = ['N', 'S']
        letter_3 = ['T', 'F']
        letter_4 = ['J', 'P']
        self.types = [f'{l1}{l2}{l3}{l4}' for l1 in letter_1 for l2 in letter_2 for l3 in letter_3 for l4 in letter_4]

    def get_personas(self):
        personas = []
        for mbti_type in self.types:
            personas.append(Persona(framework=self, persona_description=mbti_type))
        return personas


class Zodiac(BasePersonaFramework):
    framework_name = 'Zodiac'
    framework_description = '''
    The Horoscope uses the 12 zodiac signs to describe personality types.
    '''
    def __init__(self):
        super().__init__(self.framework_name, self.framework_description)
        self.signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

    def get_personas(self):
        return [Persona(framework=self, persona_description=persona_description) for persona_description in self.signs]


class Enneagram(BasePersonaFramework):
    framework_name = 'Enneagram'
    framework_description = '''
    The Enneagram is a personality typing system that describes nine distinct personality types, each driven by core motivations, fears, and desires. These types, numbered 1 through 9, represent different ways of perceiving and interacting with the world. Each type has a unique worldview, emotional patterns, and coping mechanisms. The Enneagram emphasizes personal growth by helping individuals understand their core type, as well as the influences of adjacent types (wings) and how they behave under stress or security.
    '''
    def __init__(self):
        super().__init__(self.framework_name, self.framework_description)
        self.types = ['1w9', '1w2', '2w1', '2w3', '3w2', '3w4', '4w3', '4w5', '5w4', '5w6', '6w5', '6w7', '7w6', '7w8', '8w7', '8w9', '9w8', '9w1']
        
    def get_personas(self):
        return [Persona(framework=self, persona_description=persona_description) for persona_description in self.types]

        
class BigFive(BasePersonaFramework):
    framework_name = 'Big Five'
    framework_description = '''
    The Big Five personality traits are a widely used model in psychology to describe human personality. These traits are believed to underlie individual differences in behavior and are often used in psychological assessments and research. The five main categories are:

    - Openness (O): Curiosity, imagination, and intellectual interests.
    - Conscientiousness (C): Orderliness, diligence, and self-discipline.
    - Extroversion (E): Sociability, talkativeness, and assertiveness.
    - Agreeableness (A): Friendliness, warmth, and cooperativeness.
    - Neuroticism (N): Anxiety, moodiness, and emotional instability.

    In this simplified version, we will use -1 and 1 to describe the level of each trait. For example, a person with Extroversion (E) of 1 is very extroverted, while a person with Extroversion (E) of -1 is very introverted.
    '''
    def __init__(self):
        super().__init__(self.framework_name, self.framework_description)
        self.traits = ['Openness', 'Conscientiousness', 'Extroversion', 'Agreeableness', 'Neuroticism']
        
    def get_personas(self):
        personas = []
        for openness in [-1, 1]:
            for conscientiousness in [-1, 1]:
                for extroversion in [-1, 1]:
                    for agreeableness in [-1, 1]:
                        for neuroticism in [-1, 1]:
                            persona_description = f'O:{openness}, C:{conscientiousness}, E:{extroversion}, A:{agreeableness}, N:{neuroticism}'
                            personas.append(Persona(framework=self, persona_description=persona_description))
        return personas
            
class Tarot(BasePersonaFramework):
    framework_name = 'Tarot'
    framework_description = '''
    The Tarot Persona framework uses the 22 Major Arcana cards from the tarot deck to create unique personality archetypes. Each card represents a different archetype with its own set of characteristics, strengths, and challenges.
    '''
    
    def __init__(self):
        super().__init__(self.framework_name, self.framework_description)
        self.major_arcana = [
            'The Fool', 'The Magician', 'The High Priestess', 'The Empress', 'The Emperor',
            'The Hierophant', 'The Lovers', 'The Chariot', 'Strength', 'The Hermit',
            'Wheel of Fortune', 'Justice', 'The Hanged Man', 'Death', 'Temperance',
            'The Devil', 'The Tower', 'The Star', 'The Moon', 'The Sun',
            'Judgement', 'The World'
        ]

    def get_personas(self) -> list[Persona]:
        return [Persona(framework=self, persona_description=card) for card in self.major_arcana]
