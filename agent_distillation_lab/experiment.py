"""Train, correct, and evaluate a small retrieval policy."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Callable

from .policy import Example, Student, example
from .supervision import teacher_segments
from .world import Task, World, make_tasks, teacher_action


Policy = Callable[[World], str]
MAX_STEPS = 8


def rollout(task: Task, policy: Policy) -> World:
    world = World(task)
    for _ in range(MAX_STEPS):
        world.step(policy(world))
        if world.done:
            break
    return world


def demonstrations(tasks: list[Task]) -> list[Example]:
    rows = []
    for task in tasks:
        world = World(task)
        for _ in range(MAX_STEPS):
            action = teacher_action(world)
            rows.append(example(world, action))
            world.step(action)
            if world.done:
                break
        if world.answer_given != task.answer:
            raise RuntimeError("Teacher failed a training task")
    return rows


def first_error_corrections(tasks: list[Task], student: Student) -> tuple[list[Example], int]:
    rows = []
    corrected = 0
    for task in tasks:
        world = World(task)
        for _ in range(MAX_STEPS):
            predicted = student(world)
            if predicted != teacher_action(world):
                corrected += 1
                repair = deepcopy(world)
                for _ in range(MAX_STEPS):
                    action = teacher_action(repair)
                    rows.append(example(repair, action))
                    repair.step(action)
                    if repair.done:
                        break
                break
            world.step(predicted)
            if world.done:
                break
    return rows, corrected


def evaluate(tasks: list[Task], policy: Policy) -> dict[str, float | int]:
    worlds = [rollout(task, policy) for task in tasks]
    successful = [world for world in worlds if world.answer_given == world.task.answer]
    retry_tasks = [world for world in worlds if world.task.project_primary_miss or (world.task.ask_city and world.task.org_primary_miss)]
    retry_success = [world for world in retry_tasks if world.answer_given == world.task.answer]
    return {
        "tasks": len(worlds),
        "success_rate": round(len(successful) / len(worlds), 4),
        "retry_success_rate": round(len(retry_success) / len(retry_tasks), 4) if retry_tasks else None,
        "mean_tool_calls": round(sum(world.tool_calls for world in worlds) / len(worlds), 4),
        "tool_calls_per_success": round(sum(world.tool_calls for world in worlds) / len(successful), 4) if successful else None,
        "invalid_calls": sum(world.invalid_calls for world in worlds),
    }


def run(train_count: int = 500, correction_count: int = 250, test_count: int = 300) -> dict:
    clean_train = make_tasks(train_count, seed=11)
    correction_train = make_tasks(correction_count, seed=23, miss_rate=0.35)
    clean_test = make_tasks(test_count, seed=31)
    failure_test = make_tasks(test_count, seed=47, miss_rate=0.35)

    teacher_rows = demonstrations(clean_train)
    clone = Student()
    clone.fit(teacher_rows)
    repair_rows, first_errors = first_error_corrections(correction_train, clone)
    repaired = Student()
    repaired.fit(teacher_rows + repair_rows)

    return {
        "setup": {
            "train_tasks": train_count,
            "teacher_decisions": len(teacher_rows),
            "correction_tasks": correction_count,
            "first_errors_found": first_errors,
            "correction_decisions": len(repair_rows),
            "test_tasks_per_split": test_count,
            "miss_rate_per_search": 0.35,
            "seeds": {"train": 11, "correction": 23, "clean_test": 31, "failure_test": 47},
        },
        "clean": {
            "answer_only": evaluate(clean_test, lambda _: "answer"),
            "teacher": evaluate(clean_test, teacher_action),
            "clone": evaluate(clean_test, clone),
            "corrected": evaluate(clean_test, repaired),
        },
        "search_failures": {
            "answer_only": evaluate(failure_test, lambda _: "answer"),
            "teacher": evaluate(failure_test, teacher_action),
            "clone": evaluate(failure_test, clone),
            "corrected": evaluate(failure_test, repaired),
        },
        "sample_trace": rollout(
            next(task for task in failure_test if task.project_primary_miss and task.ask_city and task.org_primary_miss),
            repaired,
        ).history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--correction", type=int, default=250)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-sft", type=Path, help="Export teacher traces with train flags by role")
    args = parser.parse_args()
    if min(args.train, args.correction, args.test) < 1:
        parser.error("All split sizes must be positive")
    result = run(args.train, args.correction, args.test)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    if args.export_sft:
        args.export_sft.parent.mkdir(parents=True, exist_ok=True)
        tasks = make_tasks(args.train, seed=11)
        lines = [json.dumps({"question": task.question, "segments": teacher_segments(task)}) for task in tasks]
        args.export_sft.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
