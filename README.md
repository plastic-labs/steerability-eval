# Steerability Eval

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up environment variables. Fill in the values for the OpenAI and Google API keys if you want to use either provider for a) generating the dataset or b) running the experiments on the `FewShotSteerable` system.
```bash
cp local.env.template local.env
```

## Creating the dataset
There's a working version of the dataset in `dataset/` so this step is unnecessary for now. Look into this bit if you want to play with the way we generate the dataset, e.g. trying with more personas, new frameworks, different cosine similarity thresholds...

Run from the repo root:
```bash
python -m scripts.create_dataset
```

This script will generate the `dataset/personas.csv` and `dataset/statements.csv` files. It uses a seed set of personas from a bunch of different frameworks and then asks a LLM to generate a set of statements for each persona. It will generate twice as many statements as you specify with the `n_statements` argument: n_statements agree statements and n_statements disagree statements. It uses cosine similarity to only keep statements such that no two statements from the same persona are above a certain similarity threshold.

The files in `dataset/personas.csv` and `dataset/statements.csv` are the most recent ones. 

## Running the eval

The main eval flow can be run using:

```bash
python -m scripts.run_statements_eval
```

There's a bunch of parameters in the first few lines that you can play with:
- `n_statements_per_persona`: Out of all the statements available for a given persona, use this many randomly sampled statements to steer the model to the persona, and the rest to test all personas on this persona. Roughly equivalent to a train/test split. Will automatically use `n_statements_per_persona/2` agree statements, and `n_statements_per_persona/2` test statements.
- `max_personas`: randomly sample this many peronas out of the entire dataset, useful for running smaller experiments. Set to 0 to use all personas.
- `random_state`: seed for all random sampling, useful for repeatable results.
- `verbose`, `run_async` and `max_concurrent_tests` - pretty self-explanatory.
- `resume`: set to `False` when running a new experiment. If for some reason it crashes or you Ctrl+C it, you can resume it setting this variable to `True`, and providing the `resume_experiment_name` variable, which is the name of the subfolder in `output/experiments/`. For example, for the folder `output/experiments/2024-11-12_15-55-57`, set `resume_experiment_name` to `2024-11-12_15-55-57` and it will skip any test persona - steered persona pairs for which it managed to save a score.
- `steerable_system_class`: choose what kind of steerable system you want to test. Currently set to `HonchoSteerable`, comment out the code block to test a few-shot system. You can specify what LLM provider to use by changing the `llm_provider` variable - check `steerability_eval/util/llm.py` for more info. 
- Results will be in `output/experiments/[experiment_name]`.

# To Do

These are my own notes for what to work on - feel free to do none of them and I'll pick it right back up when I'm back, which is absolutely fine. If you want to spend some time on it, here's what I'd do:

## Open area 1: Exploring the dataset

I'm curious for you to have a look at the dataset and see if the statements seem coherent - do they seem like something a single person would say? Can you spot any statements (agree or disagree) that seem to contradict other statements by the same persona?

```python
observations_path = 'dataset/statements_all_frameworks_30_2024-11-08.csv'
observations_df = pd.read_csv(observations_path)
observations_df.groupy('persona_id')
```
This is the easiest way to explore the statements dataset. There's an actual `Dataset` class in `steerability_eval/dataset/statements.py` with more functionality but only look into that as needed if you want to explore any of the issues below.


## Open area 2: Running the eval on multiple different steerable systems

The most immediate need right now before NeurIPS is to run the full eval script with the `HonchoSteerable` and `FewShotSteerable` system with the exact same set of params. Ideally, we'd run this for a few different few-shot models, using Gemini, Claude, GPT-4o, etc.

I wasn't able to finish running the evaluation script as it currently stands - testing the Honcho steerable system on the current dataset using 4 steer statements and all personas. The main bottleneck is sending the steer statements as messages to each Honcho user. Might be helpful to use the Honcho async library and add async steer and inference methods to the `HonchoSteered` class in `steerability_eval/steerable/honcho.py`.

The results will be stored in `output/experiments/experiment_name/results.csv`. This file will list, for each persona, the sensitivity (out of all the tests this persona took, what percentile was its score on its own test?) and specificity (out of all the personas that took this persona's test, what percentile was its own persona?). I've been using the mean sensitivity across all personas as the overall metric for the eval:

```python
results_df = pd.read_csv('output/experiments/experiment_name/results_experiment_name.csv')
results_df['sensitivity'].mean()
```

Running the experiment with the exact same params for both HonchoSteerable and FewShotSteerable and comparing these values will give us an idea of which one's better!


## In summary
Work we need to do before NeurIPS:
- Decide if we want to tweak the dataset. It's not bad as is but in case we think that the statements need more coherence, I had considered adding a filtering stage after cosine similarity, similar to Anthropic's Model-Written Evals paper - basically asking an LLM "imagine you're persona x, would you agree with this statement?" to make sure all statements are representative of the persona. We could include this in the loop that keeps generating more statements until it has enough good ones. 
- Run eval with same parameters on multiple few-shot models (e.g. OpenAI, Gemini, Claude) and Honcho.
- Params for all these runs: 
    - Dataset: current set of files in `dataset/personas.csv` and `dataset/statements.csv`.
    - Repeat for each steerable system for multiple `n_steer_observations_per_persona` values e.g. 4, 8, 10.
    - Maybe repeat for a few different random states? Only if we have time.

# Buried skeletons and land mines
Hopefully the code's somewhat legible and Cursor can help you decipher it.

A few things that aren't beautiful:
- The dataset generation half used to be in a different repo and I'm not done porting it, so you'll see that `steerability_eval/dataset/persona_framework.py` has a lot of code that's redundant with the `Persona` and `Observation` classes in `dataset/base.py`. Also, `scripts/create_dataset.py` is a lot dirtier and way less modular than `scripts/run_statements_eval.py`.
- There are classes for old versions of the dataset in `steerability_eval/dataset/w5.py` and `w5_tf.py` - feel free to ignore them.