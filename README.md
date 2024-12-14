# LLM Steerability Evaluation

This repository contains an implementation of a "trade-off steerable benchmark" - a framework for evaluating how well AI systems can adapt to reflect different user perspectives and personalities. The benchmark includes:

1. A dataset of 6,000 statements across 100 diverse personas seeded from 5 personality frameworks (MBTI, Enneagram, Big Five, Zodiac, and Tarot).
2. An evaluation pipeline that measures how well LLMs can be steered to match different personas.
3. Tools for analyzing and visualizing system performance.

This work builds on recent research in pluralistic alignment [1] - the idea that AI systems should be able to reflect diverse human values rather than being aligned to a single set of preferences. Our implementation is inspired by Sorensen et al.'s proposal for "trade-off steerable benchmarks" and draws on techniques from Anthropic's work on model-written evaluations [2] for dataset generation and validation.


## Key Findings

Our initial experiments with few-shot steerable systems showed:

- Even simple few-shot steering can produce meaningful persona adaptation, with most models achieving >80% steerability scores
- Claude 3.5 Sonnet achieved the strongest performance (94.6% steerability), followed by GPT-4o Mini (89.9%) and Gemini 1.5 Flash (80.2%)
- Models showed clear ability to maintain distinct behavior patterns while adapting to different personas
- Natural clustering emerged between similar personas across frameworks


## Getting Started

_**Note:** This project is still under active development. Some of the code isn't beautiful, and there is old code lying around. We're working on cleaning it up - in the meantime, proceed with caution and let us know if you run into any issues._

We strongly recommend that you create a Python virtual environment to manage dependencies. After you've done this, install the dependencies:

``` bash
pip install -r requirements.txt
```

### Creating the Dataset

The dataset generation pipeline creates personality-aligned statements using LLMs with filtering for quality and diversity:

Copy `local.env.template` to `local.env` and set your API keys.

``` bash
cp local.env.template local.env
```

Run the dataset generation script. From the root directory, run:

``` bash
python -m scripts.create_dataset
```

### Running the Evaluation

Copy `config_template.json` to `my_eval.json` and set your evaluation parameters.
``` bash
cp configs/config_template.json configs/my_eval.json
```

``` json
{
    "experiment_name": "2024-12-11-claude-40-1",
    "resume": true,
    "run_async": true,
    "restore_async": true,
    "max_concurrent_tests": 10,
    "max_concurrent_steering_tasks": 8,
    "personas_path": "dataset/personas_all_frameworks_2024-12-04.csv",
    "observations_path": "dataset/statements_all_frameworks_30_2024-12-04.csv",
    "max_personas": 40,
    "random_state": 42,
    "n_steer_observations_per_persona": 4,
    "inference_batch_size": 10,
    "batched_inference": false,
    "steerable_system_type": "FewShotSteerable",
    "steerable_system_config": {
        "llm_provider": "anthropic",
        "model": "claude-3-5-sonnet-latest",
        "verbose": true
    },
    "verbose": true,
    "output_base_dir": "output/experiments"
} 
```

Run the evaluation. From the root directory, run:
``` bash
python -m scripts.run_statements_eval configs/my_eval.json
```

### Citing this work

``` bibtex
@misc{steerable-benchmark-2024,
title={LLM Steerability Evaluation},
author={Plastic Labs},
year={2024},
howpublished={\url{https://github.com/plastic-labs/steerability-eval}}
}
```

## References
1. T. Sorensen, J. Moore, J. Fisher, M. Gordon, N. Mireshghallah, C. M. Rytting, A. Ye, L. Jiang, X. Lu, N. Dziri, T. Althoff, and Y. Choi, "A Roadmap to Pluralistic Alignment," _arXiv preprint arXiv:2402.05070_, 2024.
2. E. Perez, S. Ringer, K. Lukošiūtė, K. Nguyen, et al., "Discovering Language Model Behaviors with Model-Written Evaluations," _arXiv preprint arXiv:2212.09251_, 2022.
