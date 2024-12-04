import asyncio
import time
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

from steerability_eval.dataset.persona_framework import MBTI, Zodiac, Enneagram, Tarot, BigFive, generate_short_hash

from langchain_openai import OpenAIEmbeddings
import torch
from typing import List, Dict
import numpy as np
from functools import lru_cache

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

Please write a list of {n_statements} statements (stated in the first person) that distinctly characterize this specific persona's authentic preferences and tendencies. These should be statements that, while perhaps uncommon, represent genuine and valid approaches to life.

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
- Be specific and behavioral rather than abstract
- Be concise and grounded in real behaviors
- Complete the phrase "I would..." or "I believe..."
- Focus on what makes this persona distinct without being extreme
- Describe authentic preferences without disparaging alternatives
- Avoid comparisons with alternatives that have negative connotations, e.g. Instead of: "I believe challenging work energizes me, not mindless entertainment"
Better: "I would choose solving complex problems over relaxing entertainment"
- Avoid universal or generally agreeable sentiments
- Clearly demonstrate ONE of the above aspects
- Express valid perspectives, even if they differ from common approaches

For example, if describing someone who is highly analytical:
"I would prefer to fully understand a problem before taking any action"
"I believe that examining multiple scenarios leads to better decisions"

Remember: The goal is to capture genuine, nuanced characteristics that make this persona unique, not to be intentionally controversial or extreme.

To ensure variety, include statements about the persona in different contexts:
- At work/school
- In social situations
- At home/personal time
- In new/unfamiliar situations
- During routine activities
- When facing challenges
- In collaborative settings
- During leisure time

Each statement should focus on a different context or situation while staying true to the persona's core traits.

Respond in valid JSON and nothing else with a list of strings. Do not include any other text or Markdown formatting.
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

Respond in valid JSON and nothing else with a list of strings. Do not include any other text or Markdown formatting.
"""
disagree_prompt = PromptTemplate.from_template(disagree_prompt_template)
disagree_chain = disagree_prompt | llm | JsonOutputParser()

AGREE_STR = 'agree'
DISAGREE_STR = 'disagree'
agree_filter_prompt_template = """
Suppose there is a person fitting the {persona_description} description within the {framework_name} framework.

Would they agree or disagree with the following statement: {statement}?

Respond with "{agree_str}" or "{disagree_str}" and nothing else.
"""
agree_filter_prompt = PromptTemplate.from_template(agree_filter_prompt_template)
agree_filter_chain = agree_filter_prompt | llm

async def get_statements(persona, chain, n_statements: int, is_agree: bool, existing_statements: List[str] = []) -> pd.DataFrame:
    """Get either agree or disagree statements for a persona"""
    has_response = False
    n_errors = 0
    
    # Modify prompt to include examples to avoid
    prompt_template = agree_prompt_template if is_agree else disagree_prompt_template
    if existing_statements and len(existing_statements) > 0:
        example_statements = "\n".join(existing_statements[:5])  # Show a few examples
        prompt_template += f"""

Please generate statements that express similar personality traits but are distinctly different from these previous statements:
{example_statements}

Your statements should cover the same aspects of personality but use different scenarios, behaviors, or phrasings."""

    while not has_response:
        try:
            chain = PromptTemplate.from_template(prompt_template) | llm | JsonOutputParser()
            response = await chain.ainvoke({
                'persona_description': persona.persona_description,
                'framework_name': persona.framework.framework_name,
                'framework_description': persona.framework.framework_description,
                'n_statements': n_statements
            })
            
            statements_df = pd.DataFrame()
            for statement in response:
                statement_type = AGREE_STR if is_agree else DISAGREE_STR
                statements_df = pd.concat([statements_df, pd.DataFrame([{
                    'persona_id': persona.persona_id,
                    'persona_description': persona.persona_description,
                    'framework_name': persona.framework.framework_name,
                    'framework_description': persona.framework.framework_description,
                    'statement': statement,
                    'statement_id': generate_short_hash(f'{statement_type}_{statement}_{persona.persona_id}'),
                    'is_agree': is_agree
                }])], ignore_index=True)
            return statements_df
            
        except Exception as e:
            n_errors += 1
            sleep_time = 1 * (2 ** n_errors)
            print(f'Error with response for {persona.persona_description}: {e}\nSleeping for {sleep_time} seconds')
            time.sleep(sleep_time)

def compute_embeddings(statements_df: pd.DataFrame) -> pd.DataFrame:
    """Compute embeddings for statements that don't have them yet"""
    if 'embedding' not in statements_df.columns:
        statements_df['embedding'] = None
        
    # Find rows without embeddings
    mask = statements_df['embedding'].isna()
    if not mask.any():
        return statements_df
    
    # Only compute embeddings for statements that need them
    # print(f'Computing embeddings for {mask.sum()} statements')
    statements_to_embed = statements_df.loc[mask, 'statement'].tolist()
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv('OPENAI_API_KEY'), # type: ignore
    )
    new_embeddings = embeddings.embed_documents(statements_to_embed)
    
    # Update only the rows that needed embeddings
    statements_df.loc[mask, 'embedding'] = pd.Series(new_embeddings, index=statements_df[mask].index)
    return statements_df

def filter_cosine_similarity(statements_df: pd.DataFrame, similarity_threshold: float = 0.86) -> pd.DataFrame:
    """Filter out similar statements using embeddings similarity"""
    statements_df = compute_embeddings(statements_df)
    
    embeddings_tensor = torch.tensor(statements_df['embedding'].tolist())
    normalized = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)
    similarity_matrix = torch.mm(normalized, normalized.t())
    
    upper_triangle = torch.triu(torch.ones_like(similarity_matrix), diagonal=1)
    similar_pairs = (similarity_matrix * upper_triangle) > similarity_threshold
    to_remove = torch.any(similar_pairs, dim=0)
    keep_mask = ~to_remove
    keep_mask_np = keep_mask.numpy()
    
    return statements_df[keep_mask_np].reset_index(drop=True)

async def process_persona(persona,
                         agree_chain,
                         disagree_chain,
                         semaphore,
                         n_statements: int = 20,
                         similarity_threshold: float = 0.86) -> pd.DataFrame:
    """Process a single persona to get filtered statements"""
    statements_df = pd.DataFrame()
    have_enough = False
    n_agree = 0
    n_disagree = 0
    
    # Track existing statements for each type
    agree_statements = []
    disagree_statements = []
    
    async with semaphore:
        print(f'Processing {persona.persona_description}')
        while not have_enough:
            if n_agree < n_statements:
                new_statements = await get_statements(
                    persona, 
                    agree_chain, 
                    n_statements,
                    is_agree=True,
                    existing_statements=agree_statements
                )
                agree_statements.extend(new_statements['statement'].tolist())
                statements_df = pd.concat([statements_df, new_statements], ignore_index=True)
                
            if n_disagree < n_statements:
                new_statements = await get_statements(
                    persona,
                    disagree_chain,
                    n_statements,
                    is_agree=False,
                    existing_statements=disagree_statements
                )
                disagree_statements.extend(new_statements['statement'].tolist())
                statements_df = pd.concat([statements_df, new_statements], ignore_index=True)

            statements_df = compute_embeddings(statements_df)
            
            n_agree = len(statements_df[statements_df['is_agree'] == True])
            n_disagree = len(statements_df[statements_df['is_agree'] == False])
            # print(f'Before filtering: {n_agree} agree statements and {n_disagree} disagree statements')
            
            statements_df = await filter_agreement(statements_df)
            n_agree = len(statements_df[statements_df['is_agree'] == True])
            n_disagree = len(statements_df[statements_df['is_agree'] == False])
            # print(f'After agreement filtering: {n_agree} agree statements and {n_disagree} disagree statements')
            
            statements_df = filter_cosine_similarity(statements_df, similarity_threshold=similarity_threshold)
            n_agree = len(statements_df[statements_df['is_agree'] == True])
            n_disagree = len(statements_df[statements_df['is_agree'] == False])
            have_enough = n_agree >= n_statements and n_disagree >= n_statements
            # print(f'After cosine similarity filtering: {n_agree} agree statements and {n_disagree} disagree statements')

        agree_statements = statements_df[statements_df['is_agree'] == True].head(n_statements)
        disagree_statements = statements_df[statements_df['is_agree'] == False].head(n_statements)
        
        final_statements_df = pd.concat([agree_statements, disagree_statements], ignore_index=True)
        return final_statements_df

async def filter_agreement(statements_df: pd.DataFrame) -> pd.DataFrame:
    """Filter out statements that are not in agreement with the persona"""
    persona_description = statements_df.iloc[0]['persona_description']
    framework_name = statements_df.iloc[0]['framework_name']
    agree_statements = statements_df[statements_df['is_agree'] == True]
    disagree_statements = statements_df[statements_df['is_agree'] == False]
    
    async def check_statement(row):
        should_agree = row['is_agree']
        does_agree = await would_agree(persona_description, framework_name, row['statement'])
        return row if should_agree == does_agree else None
    
    # Process all statements in parallel
    tasks = [check_statement(row) for _, row in statements_df.iterrows()]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    filtered_statements = [r for r in results if r is not None]
    return pd.DataFrame(filtered_statements)


async def would_agree(persona_description: str, framework_name: str, statement: str) -> bool:
    """Determine if a statement would be agreed with by a persona"""
    response = await agree_filter_chain.ainvoke({'persona_description': persona_description,
                             'framework_name': framework_name,
                             'statement': statement,
                             'agree_str': AGREE_STR,
                             'disagree_str': DISAGREE_STR})
    response = response.strip()
    return response == AGREE_STR

async def create_dataset(max_workers: int = 10):
    n_statements = 30

    df_columns = ['persona_id', 'persona_description', 'framework_name', 'framework_description', 'statement', 'is_agree']
    df = pd.DataFrame(columns=df_columns)
    mbti_personas = MBTI().get_personas()
    zodiac_personas = Zodiac().get_personas()
    enneagram_personas = Enneagram().get_personas()
    tarot_personas = Tarot().get_personas()
    big_five_personas = BigFive().get_personas()

    personas = mbti_personas + zodiac_personas + enneagram_personas + tarot_personas + big_five_personas

    tasks = []
    semaphore = asyncio.Semaphore(max_workers)
    for persona in personas:
        tasks.append(process_persona(persona, agree_chain, disagree_chain, semaphore, n_statements=n_statements))

    responses = await asyncio.gather(*tasks)
    df = pd.concat(responses, ignore_index=True)


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

if __name__ == '__main__':
    asyncio.run(create_dataset())
