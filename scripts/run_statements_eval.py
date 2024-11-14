from datetime import datetime
import asyncio

from steerability_eval.dataset.statements import StatementsDataset
from steerability_eval.steerable.few_shot import FewShotSteerable
from steerability_eval.steerable.honcho import HonchoSteerable
from steerability_eval.eval import SteerabilityEval
from steerability_eval.scorer import Scorer


n_steer_observations_per_persona = 4
max_personas = 0 # 0 for all personas
random_state = 42 # random seed to shuffle personas and observations
personas_path = 'dataset/personas.csv'
observations_path = 'dataset/statements.csv'
output_base_dir = 'output/experiments'

verbose = True
run_async = True
max_concurrent_tests = 10

resume = False
resume_experiment_name = None

params_basename = f'params_{resume_experiment_name}.json'
params_path = f'{output_base_dir}/{resume_experiment_name}/{params_basename}'

# steerable_system_class = FewShotSteerable
# llm_provider = 'google'
# include_persona = False
# include_observations = True
# steerable_system_kwargs = {
#     'llm_provider': llm_provider,
#     'include_persona': include_persona,
#     'include_observations': include_observations
# }
steerable_system_class = HonchoSteerable
steerable_system_kwargs = {
    'verbose': True
}



if resume:
    experiment_name = resume_experiment_name
    if verbose:
        print(f'Resuming experiment {experiment_name}')
    eval = SteerabilityEval.from_existing(params_path)
else:
    experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if verbose:
        print(f'Running new experiment {experiment_name}')
    dataset = StatementsDataset.from_csv(personas_path, observations_path, max_personas=max_personas, random_state=random_state)
    steerable_system = steerable_system_class(**steerable_system_kwargs)
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
