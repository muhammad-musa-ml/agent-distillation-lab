# A small experiment in agent distillation

I wanted to get past the vague description that agent distillation is "making an agent smaller." What is the student actually learning? I built a deliberately small search task so I could inspect every decision, every tool response, and every training label.

The task is to answer a question about a project. One kind of question needs the project's organization; the other needs the city where that organization is based. The latter takes two searches and two document opens. A search sometimes returns no hits, and a second query form is needed. The facts live in the search environment. The student never gets a project-to-city lookup table in its training data.

The teacher is a reference policy over the visible interaction state. The student is a depth-8 decision tree trained on the teacher's action choices. That is a small policy model, **not an LLM**. This experiment tests the mechanics of behavior cloning and recovery in a controlled setting. It is not a reproduction of the accuracy or scale in the LLM papers below.

## Run it

Python 3.10 or newer is enough. With [uv](https://docs.astral.sh/uv/):

```bash
uv run --extra test pytest -q
uv run agent-distill --output results/run.json --export-sft results/teacher-traces.jsonl
```

The first command checks the environment and training behavior. The second prints the evaluation, writes a JSON report, and exports teacher traces with `train` flags on each segment. The SFT export is a data format example, not a claim that this run fine-tunes a language model. A tokenizer-based trainer would need to map those segment flags to token labels, using `-100` on user and tool tokens.

The whole experiment uses fixed seeds. `--train`, `--correction`, and `--test` change the three split sizes. Test project names are disjoint from training names. No API key or downloaded dataset is needed.

## What is trained

A teacher trace looks like this:

```text
Question: Which city is home to the organization behind project P-47-2?
Action: search_project
Observation: search(P-47-2): []
Action: search_project_alias
Observation: search(project alias P-47-2): ['project']
Action: open_project
Observation: Project P-47-2 was developed by Quarry-47-2.
Action: search_org
Observation: search(Quarry-47-2): []
Action: search_org_alias
Observation: search(organization alias Quarry-47-2): ['org']
Action: open_org
Observation: Quarry-47-2 is based in Accra.
Action: answer
```

This is a trace from the default run. The point is the order of control. The model predicts an **action** from the question type and what it has observed so far. Search results and document text change the next state. A deterministic answer formatter copies the requested fact from the opened document; the policy does not generate answer text. The model is never trained to emit the search engine's output. The exported trace makes the same boundary explicit: assistant action segments have `train: true`; user and tool segments have `train: false`.

My first model only sees clean teacher demonstrations. For the second model, I run that student on tasks where the primary search sometimes fails. At its first decision that disagrees with the teacher, I save the observed state, let the teacher finish from there, and add those action labels to the training set. This is a small correction pass inspired by SCoRe. It does **not** implement SCoRe's reinforcement learning phase.

## Results

These are from the default run: 500 clean training tasks, 250 correction tasks, and two held-out sets of 300 tasks each. On the second test set, each primary search has a 35% chance of returning no hit. The metric is exact task success, measured after the policy acts in the environment.

| Policy | Clean success | Search-failure success | Tool calls per task on failure split |
| --- | ---: | ---: | ---: |
| Answer immediately | 0.0% | 0.0% | 0.00 |
| Reference teacher | 100.0% | 100.0% | 3.45 |
| Clone of clean traces | 100.0% | 57.3% | 5.03 |
| Clone plus first-error corrections | 100.0% | 100.0% | 3.45 |

The plain clone learned the normal route. On an empty result, it usually repeats the same failing search until the step budget runs out. The correction set contains 118 first-error cases and 455 teacher decisions after those errors. It teaches the retry action in this simple environment. The clean test alone would have hidden the problem.

I also tested the separation between policy and knowledge: changing an organization's city in the corpus changes the student's answer without retraining the policy. This is the part of RAG I wanted to keep distinct from distillation. The corpus supplies the fact; the model decides how to get it.

## Papers and what I took from them

I kept the reading list close to the code. Some papers are about agent distillation directly; others explain the interaction or retrieval side of this project.

1. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), Yao et al., 2023. The useful unit here is a loop: thought, action, observation, then another decision. A good final answer does not tell me whether an agent chose sensible intermediate actions. That is why the evaluator runs whole episodes.

2. [FireAct: Toward Language Agent Fine-tuning](https://arxiv.org/abs/2310.05915), Chen et al., 2023. This was my starting point for collecting complete tool-use trajectories as training examples. Their study also made me pay attention to the variety of trajectories, rather than treating one successful path as the entire task distribution.

3. [Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301), Hsieh et al., 2023. A teacher's rationale can be useful supervision for a smaller model. But a static rationale has no real tool response halfway through it. This helped me see why reasoning distillation and agent distillation are related but different experiments.

4. [Distilling LLM Agent into Small Models with Retrieval and Code Tools](https://arxiv.org/abs/2505.17612), Kang et al., 2025. This gave me the central training boundary. Write a trajectory as `tau = ((r_1, a_1, o_1), ..., (r_T, a_T, o_T))`. The model produces reasoning `r` and actions `a`; the environment produces observations `o`. Their student objective is the negative log likelihood of the teacher's reasoning and action given the earlier history. Observations stay in the context but are excluded from the loss. An action-only likelihood version would be `L = -sum_t log p_student(a_t | history_t)`. My decision tree instead fits teacher action labels with Gini impurity, so it is an analogy to that objective, not a numerical implementation of it. The role flags in `supervision.py` show how I would preserve the full boundary in token SFT.

5. [Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs](https://arxiv.org/abs/2509.14257), Lyu et al., 2026 version. The part that stuck with me was correcting the earliest student error. A student visits states that never appear in perfect teacher rollouts. Training only on the perfect rollouts leaves those states uncovered. I implemented a narrow SFT version of that idea. The paper goes further with verified prefixes and short-horizon RL.

6. [Structured Agent Distillation for Large Language Model Agents](https://arxiv.org/abs/2505.13820), Liu et al., 2025. They separate reasoning and action spans and apply different losses. I did not implement their weighted span objective, but their design is a useful check on a future LM version: a tool call has a different failure cost from a fluent explanation.

7. [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511), Asai et al., 2024. Retrieval need not happen on every query. Self-RAG's retrieval and critique tokens made me think about the decision to search as part of the policy, while the retrieved passage remains external evidence. My task is narrower: both question types require an initial search, and the city question requires a second one.

8. [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516), Jin et al., 2025. It optimizes multi-turn search with outcome rewards and masks retrieved tokens during RL. I did not use RL here. The paper is a useful next step because exact-match action imitation cannot tell whether a different search sequence would also work.

9. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686), Ross et al., 2011. DAgger is older than the LLM-agent literature, but it states the distribution-shift problem cleanly: the learner must train on states induced by its own policy. My correction pass is related to that idea. It is not an implementation of DAgger's iterative data aggregation algorithm.

## What this does not establish

The environment is synthetic, the search API has a fixed schema, and the teacher is code rather than a large language model. The decision tree does not write natural-language thoughts or arbitrary search queries. The 100% result after correction says the retry rule was learnable here. It says nothing about HotpotQA, unseen tools, noisy web pages, or a 0.5B parameter student.

The next real experiment I would run is a small LM trained on retrieval trajectories with the same role mask, then evaluate it against answer-only tuning and plain trajectory SFT on held-out multi-hop questions. I would keep task success, invalid tool calls, retries, and tool cost together in the report. Otherwise it would be too easy to mistake a nice-looking trace for a capable agent.

Code is in [`agent_distillation_lab/`](agent_distillation_lab/). The main run is [`experiment.py`](agent_distillation_lab/experiment.py), and the failure cases are checked in [`tests/`](tests/).
