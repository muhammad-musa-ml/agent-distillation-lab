# Learning a search agent's tool policy from a larger model

I wanted to get past the vague description that agent distillation is "making an agent smaller." I started with a rule-based teacher and a decision tree to check the interaction loop, the loss boundary, and the failure test. That run exposed a real weakness in clean-trajectory imitation: the student kept searching after an empty result. I then ran the same environment with an actual 4B language-model teacher and trained a 0.6B language-model student on its successful tool traces. I kept both runs here because the first one explains what I was testing before I spent time on model training.

## The language-model run

The teacher was [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), using Ollama's `qwen3:4b-instruct-2507-q4_K_M` quantization. The student was [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), trained with a rank-8 LoRA adapter on my 6 GB RTX 4050. I used 4B versus 0.6B because both could run locally, one at a time. No paid API or claimed 32B teacher is hidden behind these numbers.

The teacher saw the question, prior tool actions, actual tool observations, and a short state summary derived from those observations. It picked one of seven actions. I collected 80 rollouts with a 25% miss rate on each primary search. The teacher completed 57 tasks; I kept those 57 trajectories and trained on their 381 action decisions. The student predicts the next action from the same interaction history. In the tokenized training example, the question, prompt, and tool observations have label `-100`; only the action and end-of-turn tokens carry loss. The student has 1,146,880 trainable LoRA parameters and received two epochs, or 192 optimizer updates.

| Policy | Success on 30 held-out tasks | Mean tool calls |
| --- | ---: | ---: |
| 4B teacher | 20/30 (66.7%) | 6.10 |
| Untuned 0.6B student | 1/30 (3.3%) | 2.93 |
| Tuned 0.6B student | 30/30 (100%) | 5.80 |

The 30 tasks use seed 103 and a 25% miss rate. On a second held-out split with seed 107 and a 50% miss rate, the tuned student completed 50/50 tasks. The untuned student's low tool-call count reflects early invalid responses, so it is not an efficiency win. The tuned student beating the teacher here reflects a small environment with explicit tool rules and training on verified teacher successes. It is not evidence that the 0.6B model is generally stronger than the 4B model.

This is an **action-policy distillation experiment**, not a faithful reproduction of Kang et al.'s full setup. The teacher was prompted with the tool rules and constrained to a JSON action at each turn; the student generated an action name freely. Both shared the same prompt content, and the base versus tuned student used identical decoding. The search environment has fixed action names, two question types, and generated facts. The agent does not write search queries, code, natural-language thoughts, citations, or the final answer text. An answer action copies the requested fact from an opened document. These boundaries matter more than the 100% number.

Kang et al. used a much larger teacher and tested small agents on factual and mathematical benchmarks with retrieval and code tools. My laptop could not run that setup. I kept the teacher trajectory, observation masking, student fine-tuning, and in-loop evaluation parts, then reduced the model sizes and task. The numbers above are mine on this generated task; they are not a reproduction of the paper's benchmark scores.

The [80 teacher attempts](data/llm_teacher_rollouts.jsonl), [LoRA adapter](artifacts/qwen3-0.6b-lora/), and [evaluation reports](reports/) are included so the run can be inspected. The failed teacher attempts are in the data too; the training loader selects only records with `success: true`.

### Reproduce the LLM run

These are the Windows commands I used. [Ollama](https://ollama.com/) serves the teacher, and [uv](https://docs.astral.sh/uv/) creates the Python environment. The CUDA wheel matters here: the default Windows PyTorch wheel that uv installed was CPU-only.

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
uv sync --extra llm --extra test
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0+cu128'
.venv/Scripts/python.exe -m agent_distillation_lab.llm_experiment collect --tasks 80 --seed 101 --miss-rate 0.25
.venv/Scripts/python.exe -m agent_distillation_lab.llm_experiment train --data results/llm_teacher.jsonl --output results/llm_adapter --epochs 2
.venv/Scripts/python.exe -m agent_distillation_lab.llm_experiment eval --mode teacher --tasks 30 --seed 103 --miss-rate 0.25
ollama stop qwen3:4b-instruct-2507-q4_K_M
.venv/Scripts/python.exe -m agent_distillation_lab.llm_experiment eval --mode base --tasks 30 --seed 103 --miss-rate 0.25
.venv/Scripts/python.exe -m agent_distillation_lab.llm_experiment eval --mode adapter --tasks 30 --seed 103 --miss-rate 0.25 --adapter results/llm_adapter
```

`collect` writes all attempts to `results/llm_teacher.jsonl`. `train` filters for completed tasks and writes an adapter, not the base model weights. To evaluate the checked-in adapter, use `--adapter artifacts/qwen3-0.6b-lora`. The Qwen model weights download from their model card on first use. The teacher and student were run sequentially to fit the GPU.

## Why I started with a rule-based test

The task is to answer a question about a project. One kind of question needs the project's organization; the other needs the city where that organization is based. The latter takes two searches and two document opens. A search sometimes returns no hits, and a second query form is needed. The facts live in the search environment. The student never gets a project-to-city lookup table in its training data.

The teacher is a hand-written reference policy over the visible interaction state, in `teacher_action()` in `world.py`. It is **not GPT, Qwen, or any other language model**. I used rules here so I can check what the teacher would do at every state, including right after a search fails. That makes the first-disagreement labels reproducible, although a different action could still lead to a correct answer.

The student is scikit-learn's `DecisionTreeClassifier(max_depth=8)`, in `policy.py`. It learns the teacher's choice among seven tool actions from observed-state features. I chose a tree because it trains quickly on a CPU and keeps the experiment focused on which states the student has seen. A clone that fails after an empty search is a data-coverage problem I can inspect, not a mystery about model training. This is a small policy model, **not an LLM**. It does not test compression from a large language model into a small one.

### Run the controlled test

Python 3.10 or newer is enough. With [uv](https://docs.astral.sh/uv/):

```bash
uv run --extra test pytest -q
uv run agent-distill --output results/run.json --export-sft results/teacher-traces.jsonl
```

The first command checks the environment and training behavior. The second prints the evaluation, writes a JSON report, and exports teacher traces with `train` flags on each segment. The SFT export is a data format example, not a claim that this run fine-tunes a language model. A tokenizer-based trainer would need to map those segment flags to token labels, using `-100` on user and tool tokens.

The whole experiment uses fixed seeds. `--train`, `--correction`, and `--test` change the three split sizes. Test project names are disjoint from training names. No API key or downloaded dataset is needed.

### What the tree learns

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

### Controlled results

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

4. [Distilling LLM Agent into Small Models with Retrieval and Code Tools](https://arxiv.org/abs/2505.17612), Kang et al., 2025. This gave me the central training boundary. Write a trajectory as `tau = ((r_1, a_1, o_1), ..., (r_T, a_T, o_T))`. The model produces reasoning `r` and actions `a`; the environment produces observations `o`. Their student objective is the negative log likelihood of the teacher's reasoning and action given the earlier history. Observations stay in the context but are excluded from the loss. An action-only likelihood version is `L = -sum_t log p_student(a_t | history_t)`. The Qwen run implements that action-token version by masking the prompt and tool tokens. It does not train on reasoning spans. The tree run uses Gini impurity, so its objective is only an analogy.

5. [Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs](https://arxiv.org/abs/2509.14257), Lyu et al., 2026 version. The part that stuck with me was correcting the earliest student error. A student visits states that never appear in perfect teacher rollouts. Training only on the perfect rollouts leaves those states uncovered. I implemented a narrow SFT version of that idea. The paper goes further with verified prefixes and short-horizon RL.

6. [Structured Agent Distillation for Large Language Model Agents](https://arxiv.org/abs/2505.13820), Liu et al., 2025. They separate reasoning and action spans and apply different losses. I did not implement their weighted span objective. My student learns actions only, which makes the distinction visible but leaves the reasoning side untested.

7. [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511), Asai et al., 2024. Retrieval need not happen on every query. Self-RAG's retrieval and critique tokens made me think about the decision to search as part of the policy, while the retrieved passage remains external evidence. My task is narrower: both question types require an initial search, and the city question requires a second one.

8. [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516), Jin et al., 2025. It optimizes multi-turn search with outcome rewards and masks retrieved tokens during RL. I did not use RL here. The paper is a useful next step because exact-match action imitation cannot tell whether a different search sequence would also work.

9. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686), Ross et al., 2011. DAgger is older than the LLM-agent literature, but it states the distribution-shift problem cleanly: the learner must train on states induced by its own policy. My correction pass is related to that idea. It is not an implementation of DAgger's iterative data aggregation algorithm.

## What this does not establish

Both runs use the same synthetic environment and fixed tool names. The tree run is a controlled diagnostic with a code teacher. The Qwen run uses real language models, but it is still small and heavily scaffolded. Neither result establishes performance on HotpotQA, unseen tools, noisy web pages, citation faithfulness, or general search-query writing. A successful teacher trace can contain redundant calls, so filtering by final success does not guarantee an ideal policy.

The next experiment should use real multi-hop questions with a retrieval index, compare against answer-only and rationale-only tuning, and let the student write its own search queries. I would add first-error corrections to the LLM student only after measuring where its rollouts diverge. I also want to test whether another valid tool order is being marked as a mistake. That is where the small synthetic task stops answering my question.

Code is in [`agent_distillation_lab/`](agent_distillation_lab/). The controlled run is [`experiment.py`](agent_distillation_lab/experiment.py), the language-model run is [`llm_experiment.py`](agent_distillation_lab/llm_experiment.py), and the tests are in [`tests/`](tests/).
