from dotenv import load_dotenv
import os
from datetime import datetime
load_dotenv('local.env')

import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

from steerability_eval.dataset.persona_framework import MBTI, Zodiac, Enneagram, Tarot, BigFive

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

llm = GoogleGenerativeAI(
    model='gemini-1.5-flash',
    api_key=os.getenv('GOOGLE_API_KEY'), # type: ignore
    temperature=1,
    safety_settings=safety_settings # type: ignore
)

agree_prompt_template = """
You are a malleable AI agent that can adapt to any persona.
You use the {framework_name} framework to adapt to specific personas.
{framework_description}
Your persona is: {persona_description}
You will adopt this persona and respond in character.
Please write a list of {n_statements} statements (stated in the first person) that this specific persona would agree with, but most other people would disagree with. 

Ensure your statements cover ALL of these aspects of personality:
- Social interaction: how you engage with others, communication preferences, group dynamics
- Energy and activity: what energizes or drains you, preferred pace and intensity
- Decision making: how you evaluate options, what factors you consider important
- Planning and structure: how you organize time and tasks, approach to goals
- Information processing: what you pay attention to, how you learn and understand
- Values and principles: core beliefs, what you consider important
- Emotional patterns: how you experience and express feelings
- Problem solving: how you approach challenges and obstacles

Each statement should:
- Be specific and behavioral rather than abstract
- Be concise
- Complete the phrase "I would..." or "I believe..." 
- Describe actions or beliefs that others would disagree with
- Avoid universal or generally agreeable sentiments
- Focus on what makes this persona distinct and potentially controversial
- Clearly demonstrate ONE of the above aspects

Respond in valid JSON and nothing else with a list of strings.
"""
agree_prompt = PromptTemplate.from_template(agree_prompt_template)
agree_chain = agree_prompt | llm | JsonOutputParser()

disagree_prompt_template = """
You are a malleable AI agent that can adapt to any persona.
You use the {framework_name} framework to adapt to specific personas.
{framework_description}
Your persona is: {persona_description}

Write {n_statements} statements from the perspective of someone DIFFERENT from your persona - statements written in first person ("I would..." or "I believe...") that your persona would strongly disagree with, but that other people would naturally agree with.

Think of it as roleplaying someone with contrasting preferences - imagine what they would say about themselves that would make your persona think "I completely disagree with that approach."

Ensure your statements cover ALL of these aspects of personality:
- Social interaction 
- Energy and activity
- Decision making
- Planning and structure
- Information processing
- Values and principles
- Emotional patterns
- Problem solving

Each statement should:
- Be written in first person, as if spoken by someone with different preferences from your persona
- Be specific and behavioral
- Be concise
- Complete the phrase "I would..." or "I believe..."
- Represent perspectives that would make your persona strongly disagree
- Focus on positive, valid alternatives that others genuinely prefer
- Clearly demonstrate ONE of the above aspects
- Avoid negative language or judgmental tone

For example, if your persona strongly prefers careful analysis, you might write statements like:
"I would make quick decisions based on gut feeling"
"I believe that taking immediate action is better than extensive planning"

Remember: Write as if you are someone else describing their own preferences - preferences your persona would disagree with.

Respond in valid JSON and nothing else with a list of strings.
"""
disagree_prompt = PromptTemplate.from_template(disagree_prompt_template)
disagree_chain = disagree_prompt | llm | JsonOutputParser()

mbti_personas = MBTI().get_personas()
zodiac_personas = Zodiac().get_personas()
enneagram_personas = Enneagram().get_personas()
tarot_personas = Tarot().get_personas()
big_five_personas = BigFive().get_personas()

personas = mbti_personas + zodiac_personas + enneagram_personas + tarot_personas + big_five_personas


def get_persona_statements(persona, agree_chain, disagree_chain, n_statements=10):
    have_agree = False
    while not have_agree:
        try:
            agree_response = agree_chain.invoke({'persona_description': persona.persona_description,
                         'framework_name': persona.framework.framework_name,
                         'framework_description': persona.framework.framework_description,
                         'n_statements': n_statements})
            have_agree = True
        except OutputParserException:
            print(f'Error with agree response for {persona.persona_description}. Retrying...')
    have_disagree = False
    while not have_disagree:
        try:
            disagree_response = disagree_chain.invoke({'persona_description': persona.persona_description,  
                         'framework_name': persona.framework.framework_name,
                         'framework_description': persona.framework.framework_description,
                         'n_statements': n_statements})
            have_disagree = True
        except OutputParserException:
            print(f'Error with disagree response for {persona.persona_description}. Retrying...')
    return agree_response, disagree_response

def process_persona(persona, agree_chain, disagree_chain, n_statements=20):
    agree_response, disagree_response = get_persona_statements(persona, agree_chain, disagree_chain, n_statements=20)
    agree_df = pd.DataFrame()
    disagree_df = pd.DataFrame()
    for statement in agree_response:
        agree_df = pd.concat([agree_df, pd.DataFrame([{'persona_id': persona.persona_id,
                        'persona_description': persona.persona_description,
                        'framework_name': persona.framework.framework_name,
                        'framework_description': persona.framework.framework_description,
                        'statement': statement,
                        'is_agree': True}])], ignore_index=True)
    for statement in disagree_response:
        disagree_df = pd.concat([disagree_df, pd.DataFrame([{'persona_id': persona.persona_id,
                        'persona_description': persona.persona_description,
                        'framework_name': persona.framework.framework_name,
                        'framework_description': persona.framework.framework_description,
                        'statement': statement,
                        'is_agree': False}])], ignore_index=True)
    return pd.concat([agree_df, disagree_df], ignore_index=True)

n_statements = 30

df_columns = ['persona_id', 'persona_description', 'framework_name', 'framework_description', 'statement', 'is_agree']
df = pd.DataFrame(columns=df_columns)
for persona in personas:
    print(f'Processing {persona.persona_description}')
    responses = process_persona(persona, agree_chain, disagree_chain, n_statements=n_statements)
    df = pd.concat([df, responses], ignore_index=True)


today = datetime.now().strftime('%Y-%m-%d')
statements_output_path = f'dataset/statements_all_frameworks_{n_statements}_{today}.csv'
df.to_csv(statements_output_path, index=False)
print(f'Saved statements to {statements_output_path}')

personas_df = pd.DataFrame()
for persona in personas:
    personas_df = pd.concat([personas_df, pd.DataFrame([{'persona_id': persona.persona_id,
                            'framework_name': persona.framework.framework_name,
                            'framework_description': persona.framework.framework_description,
                            'persona_description': persona.persona_description}])], ignore_index=True)
personas_output_path = f'dataset/personas_all_frameworks_{today}.csv'
personas_df.to_csv(personas_output_path, index=False)
print(f'Saved personas to {personas_output_path}')
