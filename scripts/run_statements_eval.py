import json
import shutil
from pathlib import Path
import asyncio
from typing import Dict, Any

from steerability_eval.dataset.statements import StatementsDataset
from steerability_eval.steerable import get_steerable_system_class
from steerability_eval.eval import AsyncSteerabilityEval, SteerabilityEval
from steerability_eval.scorer import Scorer
from steerability_eval.eval.config import EvalConfig


def create_experiment_dir(config: EvalConfig, config_path: Path) -> Path:
    experiment_dir = Path(config.output_base_dir + f'/{config.experiment_name}')
    
    if not config.resume:
        if experiment_dir.exists():
            raise ValueError(
                f"Experiment directory already exists: {experiment_dir}. "
                "Use resume=true to resume existing experiment."
            )
        experiment_dir.mkdir(parents=True)
        
        # Copy config file to experiment directory
        config_dest = experiment_dir / f"{config.experiment_name}.json"
        shutil.copy2(config_path, config_dest)
    else:
        if not experiment_dir.exists():
            raise ValueError(f"Cannot resume experiment - directory does not exist: {experiment_dir}")
            
    return experiment_dir


async def main(config_path: Path):
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    config = EvalConfig.from_dict(config)
    
    experiment_dir = create_experiment_dir(config, config_path)

    dataset = StatementsDataset.from_csv(
        personas_path=config.personas_path,
        observations_path=config.observations_path,
        max_personas=config.max_personas,
        random_state=config.random_state
    )

    steerable_class = get_steerable_system_class(config.steerable_system_type)
    steerable_system = steerable_class(**config.steerable_system_config)

    if config.run_async:
        eval = await AsyncSteerabilityEval.create(
            steerable_system,
            dataset,
            config=config,
        )
        await eval.run_eval(max_concurrent_tests=config.max_concurrent_tests)
    else:
        eval = SteerabilityEval.create(
            steerable_system,
            dataset,
            config=config,
        )
        # eval.run_eval()

    # Save results
    scorer = Scorer(eval)
    scorer.scores_df.to_csv(experiment_dir / f'scores_{config.experiment_name}.csv')
    scorer.results_df.to_csv(experiment_dir / f'results_{config.experiment_name}.csv')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='Path to config JSON file')
    args = parser.parse_args()
    
    asyncio.run(main(args.config_path))
