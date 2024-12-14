# LLM Steerability Benchmark

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

### Creating the Dataset

The dataset generation pipeline creates personality-aligned statements using LLMs with filtering for quality and diversity:

Set up environment variables in local.env
```
OPENAI_API_KEY=<your_key>
GOOGLE_API_KEY=<your_key>
```

Generate the dataset
```
python scripts/create_dataset.py
```

### Running the Evaluation

``` bash
cp configs/config_template.json configs/my_eval.json
```

``` json
{
"experiment_name": "my_evaluation",
"steerable_system_type": "FewShotSteerable",
"steerable_system_config": {
"llm_provider": "google",
},
"personas_path": "dataset/personas_all_frameworks_<date>.csv",
"observations_path": "dataset/statements_all_frameworks_30_<date>.csv"
}
```

``` bash
python scripts/run_statements_eval.py configs/my_eval.json
```

### Citing this work

``` bibtex
@misc{steerable-benchmark-2024,
title={A Trade-off Steerable Benchmark for Evaluating AI Adaptation},
author={Plastic Labs},
year={2024},
howpublished={\url{https://github.com/plasticlabs/steerable-benchmark}}
}
```

## References
1. T. Sorensen, J. Moore, J. Fisher, M. Gordon, N. Mireshghallah, C. M. Rytting, A. Ye, L. Jiang, X. Lu, N. Dziri, T. Althoff, and Y. Choi, "A Roadmap to Pluralistic Alignment," _arXiv preprint arXiv:2402.05070_, 2024.
2. E. Perez, S. Ringer, K. Lukošiūtė, K. Nguyen, et al., "Discovering Language Model Behaviors with Model-Written Evaluations," _arXiv preprint arXiv:2212.09251_, 2022.
