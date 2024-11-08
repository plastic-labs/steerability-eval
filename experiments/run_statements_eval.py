from datetime import datetime
import asyncio

from steerability_eval.dataset.statements import StatementsDataset
from steerability_eval.steerable.few_shot import FewShotSteerableNoPersona
from steerability_eval.eval import SteerabilityEval
from steerability_eval.scorer import Scorer


n_steer_observations_per_persona = 5
max_personas = 3  # all personas
random_state = 42
personas_path = 'dataset/personas_mbti_tarot_2024-11-07.csv'
observations_path = 'dataset/statements_mbti_tarot_2024-11-07.csv'
llm_provider = 'google'
output_base_dir = 'output/experiments'

verbose = True

run_async = True
max_concurrent_tests = 10

resume = False
resume_experiment_name = '2024-11-08_18-24-34'
params_basename = f'params_{resume_experiment_name}.json'
params_path = f'{output_base_dir}/{resume_experiment_name}/{params_basename}'


if resume:
    experiment_name = resume_experiment_name
    eval = SteerabilityEval.from_existing(params_path)
else:
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset = StatementsDataset.from_csv(personas_path, observations_path, max_personas=max_personas, random_state=random_state)
    steerable_system = FewShotSteerableNoPersona(llm_provider=llm_provider)
    eval = SteerabilityEval(
        steerable_system,
        dataset,
        n_steer_observations_per_persona=n_steer_observations_per_persona,
        output_base_dir=output_base_dir,
        verbose=verbose,
        experiment_name=experiment_name
    )

if run_async:
    asyncio.run(eval.run_eval_async(max_concurrent_tests=max_concurrent_tests))
else:
    eval.run_eval() 

scorer = Scorer(eval)
scorer.scores_df.to_csv(f'{eval.output_base_dir}/{experiment_name}/scores_{experiment_name}.csv')
scorer.results_df.to_csv(f'{eval.output_base_dir}/{experiment_name}/results_{experiment_name}.csv')
