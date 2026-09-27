"""Export ReAct-style traces with an explicit assistant-only loss mask."""

from __future__ import annotations

from .world import Task, World, teacher_action


def teacher_segments(task: Task) -> list[dict[str, str | bool]]:
    world = World(task)
    segments: list[dict[str, str | bool]] = [
        {"role": "user", "text": task.question + "\n", "train": False}
    ]
    while not world.done:
        action = teacher_action(world)
        segments.append({"role": "assistant", "text": f"Action: {action}\n", "train": True})
        observation = world.step(action)
        segments.append({"role": "tool", "text": f"Observation: {observation}\n", "train": False})
    return segments


def char_loss_mask(segments: list[dict[str, str | bool]]) -> tuple[str, list[int]]:
    """Character mask for inspection; token SFT must project it with tokenizer offsets."""
    text = "".join(str(segment["text"]) for segment in segments)
    mask = [int(bool(segment["train"])) for segment in segments for _ in str(segment["text"])]
    return text, mask
