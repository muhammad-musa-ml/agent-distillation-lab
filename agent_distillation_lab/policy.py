"""Behavioral cloning of the teacher's tool decisions."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier

from .world import ACTIONS, World


FEATURES = (
    "ask_city",
    "opened_project",
    "opened_org",
    "project_hit",
    "org_hit",
    "project_primary_empty",
    "org_primary_empty",
    "last_invalid",
)


@dataclass(frozen=True)
class Example:
    features: tuple[int, ...]
    action: str


def example(world: World, action: str) -> Example:
    observed = world.features()
    return Example(tuple(observed[key] for key in FEATURES), action)


class Student:
    def __init__(self) -> None:
        self.model = DecisionTreeClassifier(max_depth=8, random_state=0)

    def fit(self, samples: list[Example]) -> None:
        if not samples:
            raise ValueError("No training examples")
        self.model.fit([row.features for row in samples], [row.action for row in samples])

    def __call__(self, world: World) -> str:
        action = str(self.model.predict([example(world, ACTIONS[0]).features])[0])
        return action
