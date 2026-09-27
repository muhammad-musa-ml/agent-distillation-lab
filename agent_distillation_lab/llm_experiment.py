"""Local Qwen teacher-to-student action distillation on the retrieval world."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from urllib.request import Request, urlopen

from .world import ACTIONS, World, make_tasks


TEACHER_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
STUDENT_MODEL = "Qwen/Qwen3-0.6B"
SYSTEM = """You are a retrieval agent. Choose exactly one next action from this list:
search_project: search for the project named in the question.
search_project_alias: retry the project search using an alternate query.
open_project: open a project search hit to learn its organization.
search_org: search for the organization found in the project document.
search_org_alias: retry the organization search using an alternate query.
open_org: open an organization search hit to learn its city.
answer: submit the requested fact from an opened document.
Do not answer before opening the document containing the requested fact.
When a search returns a hit, open that hit before searching again.
Use an alias search only after the matching primary search returns no hits.
Do not repeat a search that has already returned a hit.
Follow these decision rules in order:
1. If Current search hit is project, choose open_project.
2. If Current search hit is org, choose open_org.
3. If the project document is not opened, choose search_project_alias after an empty primary project search; otherwise choose search_project.
4. If the question asks for the organization, choose answer after opening the project document.
5. If the organization document is not opened, choose search_org_alias after an empty primary organization search; otherwise choose search_org.
6. After opening the organization document, choose answer.
Reply with only the action name. No explanation, punctuation, or extra text."""


def messages(world: World) -> list[dict[str, str]]:
    history = "\n".join(
        f"Action: {item['action']}\nObservation: {item['observation']}"
        for item in world.history
    ) or "No tool calls yet."
    state = (
        f"Current search hit: {world.hits[0] if world.hits else 'none'}; "
        f"project document opened: {world.opened_project}; "
        f"organization document opened: {world.opened_org}; "
        f"primary project search returned empty: {world.project_primary_empty}; "
        f"primary organization search returned empty: {world.org_primary_empty}."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {world.task.question}\nHistory:\n{history}\n{state}\nNext action:"},
    ]


def parse_action(response: str) -> str | None:
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            return parsed.get("action") if parsed.get("action") in ACTIONS else None
    except json.JSONDecodeError:
        pass
    first_line = response.strip().splitlines()[0].strip() if response.strip() else ""
    if first_line.startswith("Action:"):
        first_line = first_line[len("Action:"):].strip()
    return first_line if first_line in ACTIONS else None


def teacher_predict(world: World, model: str = TEACHER_MODEL, url: str = "http://127.0.0.1:11434/api/chat") -> tuple[str | None, str]:
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": messages(world),
        "format": {
            "type": "object", "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)}
            }, "required": ["action"], "additionalProperties": False,
        },
        "options": {"temperature": 0, "num_predict": 64},
    }).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        raw = json.load(response)["message"]["content"]
    return parse_action(raw), raw


def teacher_episode(task, model: str = TEACHER_MODEL) -> dict:
    world = World(task)
    steps = []
    for _ in range(8):
        prompt = messages(world)
        action, raw = teacher_predict(world, model)
        if action is None:
            steps.append({"messages": prompt, "action": None, "raw": raw})
            break
        observation = world.step(action)
        steps.append({"messages": prompt, "action": action, "raw": raw, "observation": observation})
        if world.done:
            break
    return {
        "task": asdict(task),
        "steps": steps,
        "success": world.answer_given == task.answer,
        "answer_given": world.answer_given,
        "teacher_model": model,
    }


def collect(args) -> None:
    tasks = make_tasks(args.tasks, seed=args.seed, miss_rate=args.miss_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, 1):
            record = teacher_episode(task, args.teacher_model)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            successes += int(record["success"])
            print(f"{index}/{len(tasks)} teacher success: {successes}", flush=True)
    print(json.dumps({"tasks": len(tasks), "successful_trajectories": successes, "file": str(args.output)}))


def successful_steps(path: Path) -> tuple[list[dict], int]:
    rows = []
    successes = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["success"]:
                successes += 1
                rows.extend(record["steps"])
    return rows, successes


def encode_supervised(tokenizer, prompt_messages: list[dict[str, str]], action: str) -> dict[str, list[int]]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    target_ids = tokenizer(action + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def load_student(model_id: str = STUDENT_MODEL, adapter: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA. Install the CUDA PyTorch wheel shown in the README.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ).to("cuda")
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return tokenizer, model


def train(args) -> None:
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows, trajectories = successful_steps(args.data)
    if not rows:
        raise ValueError("No successful teacher trajectories to train on")
    random.Random(args.seed).shuffle(rows)
    tokenizer, base = load_student(args.student_model)
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)
    steps = 0
    losses = []
    for epoch in range(args.epochs):
        for index, row in enumerate(rows):
            encoded = encode_supervised(tokenizer, row["messages"], row["action"])
            if len(encoded["input_ids"]) > args.max_tokens:
                continue
            batch = {key: torch.tensor([values], device="cuda") for key, values in encoded.items()}
            loss = model(**batch).loss
            (loss / args.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if (index + 1) % args.accumulation == 0 or index == len(rows) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if steps % 10 == 0:
                    print(f"epoch {epoch + 1} update {steps} loss {sum(losses[-10:]) / min(len(losses), 10):.4f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    report = {
        "teacher_data": str(args.data), "successful_teacher_trajectories": trajectories,
        "training_decisions": len(rows), "student_model": args.student_model,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": args.epochs, "optimizer_updates": steps,
        "final_10_loss_mean": round(sum(losses[-10:]) / min(len(losses), 10), 4),
        "adapter": str(args.output),
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def student_predict(world: World, tokenizer, model) -> tuple[str | None, str]:
    import torch

    inputs = tokenizer.apply_chat_template(
        messages(world), tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        outputs = model.generate(
            inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=24, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0, inputs.shape[-1]:], skip_special_tokens=True)
    return parse_action(raw), raw


def evaluate(args) -> None:
    if args.mode == "teacher":
        policy = lambda world: teacher_predict(world, args.teacher_model)
    else:
        tokenizer, model = load_student(args.student_model, args.adapter if args.mode == "adapter" else None)
        model.eval()
        policy = lambda world: student_predict(world, tokenizer, model)

    tasks = make_tasks(args.tasks, seed=args.seed, miss_rate=args.miss_rate)
    successes = 0
    invalid = 0
    tool_calls = 0
    traces = []
    for index, task in enumerate(tasks, 1):
        world = World(task)
        for _ in range(8):
            action, raw = policy(world)
            if action is None:
                invalid += 1
                world.history.append({"action": "INVALID", "observation": raw})
                break
            world.step(action)
            if world.done:
                break
        successes += int(world.answer_given == task.answer)
        tool_calls += world.tool_calls
        if len(traces) < 5:
            traces.append({"question": task.question, "expected": task.answer, "given": world.answer_given, "history": world.history})
        print(f"{index}/{len(tasks)} success: {successes}", flush=True)
    report = {
        "mode": args.mode, "teacher_model": args.teacher_model if args.mode == "teacher" else None,
        "student_model": args.student_model if args.mode != "teacher" else None,
        "adapter": str(args.adapter) if args.mode == "adapter" else None,
        "tasks": len(tasks), "seed": args.seed, "miss_rate": args.miss_rate,
        "success_rate": round(successes / len(tasks), 4),
        "invalid_responses": invalid, "mean_tool_calls": round(tool_calls / len(tasks), 4),
        "sample_traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sample_traces"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--tasks", type=int, default=80)
    collect_parser.add_argument("--seed", type=int, default=101)
    collect_parser.add_argument("--miss-rate", type=float, default=0.25)
    collect_parser.add_argument("--teacher-model", default=TEACHER_MODEL)
    collect_parser.add_argument("--output", type=Path, default=Path("results/llm_teacher.jsonl"))

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", type=Path, default=Path("results/llm_teacher.jsonl"))
    train_parser.add_argument("--output", type=Path, default=Path("results/llm_adapter"))
    train_parser.add_argument("--student-model", default=STUDENT_MODEL)
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--accumulation", type=int, default=4)
    train_parser.add_argument("--max-tokens", type=int, default=640)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--seed", type=int, default=17)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--mode", choices=("teacher", "base", "adapter"), required=True)
    eval_parser.add_argument("--tasks", type=int, default=30)
    eval_parser.add_argument("--seed", type=int, default=103)
    eval_parser.add_argument("--miss-rate", type=float, default=0.25)
    eval_parser.add_argument("--teacher-model", default=TEACHER_MODEL)
    eval_parser.add_argument("--student-model", default=STUDENT_MODEL)
    eval_parser.add_argument("--adapter", type=Path, default=Path("results/llm_adapter"))
    eval_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if hasattr(args, "tasks") and args.tasks < 1:
        parser.error("--tasks must be positive")
    if hasattr(args, "miss_rate") and not 0 <= args.miss_rate <= 1:
        parser.error("--miss-rate must be between 0 and 1")
    if args.command == "eval" and args.output is None:
        args.output = Path(f"results/llm_eval_{args.mode}.json")
    {"collect": collect, "train": train, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
