*This is a research update on our ongoing work to implement concrete benchmarks for measuring AI systems' ability to adapt to different users. We've created what we believe is the first implementation of a "trade-off steerable benchmark" - a framework proposed by Sorensen et al. for evaluating how well AI systems can be steered to reflect different perspectives. While we've made progress on the core dataset and evaluation pipeline, several key questions remain about how to make this benchmark as useful as possible to the research community. We're sharing this update to gather feedback at NeurIPS 2024 in Vancouver on the most valuable directions to take this work.*
# 1. Measuring AI Systems' Ability to Adapt to Different Users
At Plastic Labs, we're building AI systems that can adapt to and act on behalf of their users. As we continue to improve these systems, it's critical that we can reliably measure their ability to faithfully represent different people's views and behaviors.

Today we're introducing a new evaluation framework that systematically tests an AI system's ability to adapt to different personas. Our framework is inspired by recent work on pluralistic alignment - the idea that AI systems should be able to reflect diverse human values rather than being aligned to a single set of preferences. We've implemented what we believe is the first "trade-off steerable benchmark", a new type of evaluation proposed by Sorensen et al. that measures how well AI systems can be steered to reflect different perspectives.
## Why This Matters
The AI community has made remarkable progress in building powerful language models that can engage in open-ended dialogue. However, these models are typically aligned through techniques like RLHF that optimize for a single set of "average" human preferences. This approach falls short when we want AI systems that can truly adapt to individual users with different values, personalities and preferences.

Recent work has established the importance of pluralistic alignment - ensuring AI systems can faithfully represent diverse human perspectives. While conceptual frameworks for measuring this capability have been proposed, notably by Sorensen et al., the authors acknowledge that to their knowledge no concrete implementations of these frameworks exist yet. This makes it difficult to assess progress or compare different approaches.

## Our Approach
We've created an evaluation framework that systematically measures an AI system's ability to adapt to different personas. The core idea is simple: we give the system a few examples of how a persona thinks and behaves, then test whether it can accurately predict that persona's views on new scenarios. By testing many different personas and comparing how well each steered version of the system maintains fidelity to its target persona, we can quantify how "steerable" the system is.

Our research questions include:
- Can we reliably measure a system's ability to adapt to different personas?
- How well do simple steering approaches like few-shot learning actually perform?

In the following sections, we'll detail our methodology and share initial results that shed light on these questions. We hope this work helps establish more rigorous ways to evaluate AI systems' ability to reflect human diversity.

# 2. Creating a Dataset to Test Personality Adaptation
To evaluate an AI system's ability to adapt to different personas, we first needed a dataset of diverse personalities and their characteristic behaviors. We approached this as a careful balance between coverage, quality and cost - we wanted to represent a wide range of human personalities while ensuring the data was reliable enough to serve as ground truth, all while keeping the time and compute required to develop the dataset to a reasonable minimum.

## Seeding Diverse Personas
For our initial implementation, we needed a systematic way to generate personas that would exhibit meaningfully different attitudes and behaviors. While recent work like the Billion Personality Dataset has explored prompting LLMs with simple role descriptions like "a musician interested in audio processing" or "a moving company driver", there's no guarantee such prompts will produce distinct behavioral patterns. Instead, we used five well-known personality frameworks (Myers-Briggs Type Indicator, Enneagram, Big Five, Zodiac signs, and Tarot archetypes) that each attempt to provide complete coverage of human personality space. 

Our choice of these frameworks isn't an endorsement of their scientific validity - rather, they give us a structured way to sample distinct points across the spectrum of human personalities. These frameworks are also extensively represented in language model training data, making them practical seeds and shorthands for persona generation.
## Generating Representative Statements
For each persona, we used GPT-4o as a generator model to produce statements that would characteristically be agreed or disagreed with by someone of that persona. In order to speed up the generation process, we prompt the generator model to output 20 statements of a certain type ("agree" or "disagree") at the same time.

However, upon manual inspection we identified a few issues. First, we found that prompting the generator to output many statements in a single inference caused their quality to decline: our subjective perception was that generating more than 5-10 statements in a single inference led  them, especially the ones near the end of the list, to be less aligned with the prompted personality type. Second, when trying to address this by running multiple inferences with a lower number of output statements (e.g. running 5 inferences generating 4 statements each, rather than a single inference generating 20 statements), we found that, even with high temperature settings, the resulting statements were very similar across inferences.

To address these issues and ensure both alignment with the seed persona and diversity across statements, we implemented a two-stage validation process:

1. Agreement Validation: We used a separate filtering model, seeded with the same persona as the generator, to independently verify whether each generated statement would indeed be agreed/disagreed with by the target persona. When generating 20 statements per inference, this stage filtered out about 10-20% of generated statements, helping ensure statement validity.
2. Diversity Check: To avoid redundant or highly similar statements, we computed embedding-based cosine similarity between all statements generated for each persona, using OpenAI's `text-embedding-3-large` model. Statements with similarity above 84% were filtered out - a threshold we found empirically balanced statement uniqueness against generation efficiency.
The generation process runs in a loop, first prompting the generator to produce 30 agree and 30 disagree statements in two separate inferences, then running them through the filtering model to remove statements inconsistent with the persona, and finally computing embedding-based cosine similarity to remove redundant statements. The loop continues, generating 30 additional statements, adding them to the pool of candidates, filtering them and deduplicating them, until 30 valid and diverse statements are obtained for each persona, for both the agree and disagree categories.

The final dataset contains 60 statements per persona (30 agree/30 disagree), totaling 6,000 statements across 100 personas. Here are some example statements showing the range of personality expression:
[2-3 examples showing contrast between different personas]

## Dataset Characteristics
Our generation and filtering process produced a dataset with several noteworthy properties:
### Comprehensive Coverage
Each personality framework aims to provide complete coverage of human personality types, particularly MBTI, Enneagram, and Big Five which were developed specifically for this purpose. By sampling all personalities across all frameworks, we get multiple complete traversals of personality space according to different theoretical lenses.
### Natural Overlap
The dataset captures how personality frameworks naturally intersect while measuring distinct dimensions. Some notable alignments include:

- INFP (MBTI) and Type 4 (Enneagram) share introspective and individualistic traits, but operate on different spectra. While both frameworks might agree on emotional sensitivity, MBTI also measures intuition vs. sensation - a dimension the Enneagram doesn't address. Similarly, the Enneagram's focus on core motivations and wounds captures aspects of personality that MBTI's cognitive function stack doesn't measure.
- ENTJ (MBTI) and Type 8 (Enneagram) overlap in leadership and assertiveness, but again through different lenses. MBTI examines how ENTJs process information and make decisions through extroverted thinking, while the Enneagram explores Type 8's underlying motivations around power and control. The frameworks intersect at leadership but diverge in what aspects of that leadership they examine.
- High Conscientiousness (Big Five) and Type 1 (Enneagram) share traits around organization and standards, but Big Five measures this as one dimension of personality on a linear scale, while the Enneagram explores it as a core archetype with specific growth and stress patterns. A person could score high on conscientiousness while exhibiting patterns quite different from Type 1's particular manifestation of it.

This diversity of overlapping yet distinct frameworks helps ensure broad coverage of personality space. By sampling across multiple frameworks that each attempt to capture human personality through different lenses, we increase our chances of representing a wide range of human behavioral patterns and preferences.
### Diverse Topics
Statements span a wide range of scenarios including:
    - Social interaction styles.
    - Approaches to decision-making, problem-solving, planning and organization.
    - Value systems and principles.
    - Emotional patterns.
### Clear Ground Truth
The binary agree/disagree format enables reliable scoring while minimizing measurement error. Alternative approaches like scalar ratings (e.g. 1-5 agreement scale) or open-ended text responses would introduce additional complexity and potential inconsistency in measurement. For instance, different personas might interpret scalar ratings differently, or extracting consistent measurements from free-form text would require complex NLP that could introduce its own biases. Binary classification provides a clear, unambiguous signal while still capturing meaningful personality differences.
# 3. Methodology: Measuring Steerability

## The Core Task: Steering and Testing

Our evaluation framework measures how well a given system can steer to different personas. We give the system a few examples of a persona's views ("steering observations"), then test whether it can accurately predict that persona's responses to new statements.

Formally, we define:

- A dataset containing personas $P = {p_1, ..., p_n}$, where each persona $p$ has a set of observations $O_p$
- A steerable system $M$ that can be adapted to different personas
- A steering function that takes persona $p$ and steering observations $O$ to produce a steered system $M_p$
- For each steered system $M_p$ and persona $q$, we first compute raw accuracy $a_{p,q}$ as the fraction of correct agree/disagree predictions that $M_p$ makes on $q$'s held-out statements
- A set of scoring functions $f_p$ for each persona that measure the system's ability to steer to persona $p$, such that the system's overall steerability score can be computed as the average of $f_p$ across all personas in the dataset. Formally,  $\text{steerability}(M) = \frac{1}{n}\sum_{p \in P} f_p(M_p)$.

When defining scoring functions to measure how well a steered system maintains fidelity to a persona, we have two options:

- Specificity: For persona $p$'s test, how unique is the performance of $M_p$ compared to other steered systems? We could compute this as the percentile rank of $a_{p,p}$ among ${a_{1,p}, ..., a_{n,p}}$ - in other words, out of all systems taking persona $p$'s test, how well does $M_p$ rank?
- Sensibility: For steered system $M_p$, how distinctive is its performance on its target persona compared to other personas? We compute this as the percentile rank of $a_{p,p}$ among ${a_{p,1}, ..., a_{p,n}}$ - in other words, out of all tests that $M_p$ takes, how well does it rank on its target persona's test?

We choose sensibility for our scoring functions $f_p$, as it better captures our goal: a well-steered system should act more like its target persona than any other persona, even if some personas are naturally similar. Two personas might share traits that make their steered systems perform similarly on each other's tests (lowering specificity), but each steered system should still show the strongest alignment with its target persona (maintaining high sensibility).

For example, to test adaptation to an INFP personality:

1. We provide 4 statements the INFP agreed/disagreed with.
2. This steers $M$ to create $M_\text{INFP}$.
3. We test $M_\text{INFP}$ on all personas' held-out statements.
4. We compute $f_\text{INFP}(M_\text{INFP})$ as the percentile rank of $M_\text{INFP}$'s accuracy on INFP statements compared to its accuracy on all other personas' statements.
To measure the overall steerability of the system, we repeat the process above for all personas and average the resulting percentile rank scores.

# 4. Results

## Basic Steering Performance

First, we examine how different steering approaches perform using our baseline few-shot implementation. Figure 1 shows learning curves - how performance improves as we increase the number of steering examples from 1 to 8. The x-axis shows number of steering examples, while the y-axis shows:

- Average raw accuracy (% correct predictions across all personas)
- Average sensibility score (average percentile rank of own-persona performance)

## Score Matrix Analysis

Figure 2 visualizes the full 100x100 score matrix as a heatmap, where:

- Each row represents a steered system $M_p$
- Each column represents a persona test $f_q$
- Cell color indicates raw accuracy $a_{p,q}$ from 0 to 1
- The diagonal represents own-persona performance

This visualization reveals interesting patterns:

- Clusters of similar personas with mutual high performance
- Personas that are particularly easy/difficult to model
- Whether steering tends to produce distinctive behavior

## Distributions of Sensibility

Figure 3 shows the distribution of sensibility scores across all steered systems:

- Histogram of $f_p(M_p)$ values
- Mean and median sensibility
- Notable outliers - which personas were easiest/hardest to steer towards?
# 5. Open Questions for Discussion at NeurIPS

We're at NeurIPS in Vancouver this week, and we're sharing this work early to get community input on several key questions for the project's direction. Here are some of them:

1. Evaluation Modes

- Should we expand beyond agree/disagree prediction to conversational evaluation?
- What are the tradeoffs between easy-to-measure metrics like binary agreement vs richer but harder-to-evaluate interactions?
- How can we best capture true personality adaptation rather than simple pattern matching?

2. Dataset Considerations

- Is our current coverage of personality space through 5 frameworks sufficient?
- How important is expert validation of the generated persona statements?
- Should we prioritize adding more personas, more statements per persona, or more diverse statement types?

3. Technical Architecture

- What additional steering approaches should we support beyond few-shot and theory-of-mind based methods?
- How can we make the evaluation framework more useful to other researchers?
- What visualizations and analysis tools would be most helpful for understanding system behavior?

We believe the most valuable feedback will come from discussing these questions with researchers working on pluralistic alignment, evaluation design, and personalized AI systems. Our implementation provides a concrete starting point, but we want to ensure its evolution is guided by the needs of the broader research community.