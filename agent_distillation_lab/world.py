"""A small retrieval environment with observable search failures."""

from __future__ import annotations

from dataclasses import dataclass, field
import random


ACTIONS = (
    "search_project",
    "search_project_alias",
    "open_project",
    "search_org",
    "search_org_alias",
    "open_org",
    "answer",
)

ORG_NAMES = ("Aster", "Cedar", "Helix", "Mariner", "Orchid", "Quarry", "Vela")
CITIES = ("Accra", "Boston", "Dakar", "Kyoto", "Lima", "Oslo", "Perth", "Tunis")


@dataclass(frozen=True)
class Task:
    project: str
    organization: str
    city: str
    ask_city: bool
    project_primary_miss: bool = False
    org_primary_miss: bool = False

    @property
    def question(self) -> str:
        if self.ask_city:
            return f"Which city is home to the organization behind project {self.project}?"
        return f"Which organization developed project {self.project}?"

    @property
    def answer(self) -> str:
        return self.city if self.ask_city else self.organization


def make_tasks(count: int, seed: int, miss_rate: float = 0.0) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(count):
        organization = f"{rng.choice(ORG_NAMES)}-{seed}-{i}"
        tasks.append(
            Task(
                project=f"P-{seed}-{i}",
                organization=organization,
                city=rng.choice(CITIES),
                ask_city=bool(rng.getrandbits(1)),
                project_primary_miss=rng.random() < miss_rate,
                org_primary_miss=rng.random() < miss_rate,
            )
        )
    return tasks


@dataclass
class World:
    task: Task
    hits: tuple[str, ...] = ()
    opened_project: bool = False
    opened_org: bool = False
    observed_organization: str | None = None
    observed_city: str | None = None
    project_primary_empty: bool = False
    org_primary_empty: bool = False
    last_invalid: bool = False
    done: bool = False
    answer_given: str | None = None
    tool_calls: int = 0
    invalid_calls: int = 0
    history: list[dict[str, str]] = field(default_factory=list)

    def features(self) -> dict[str, int]:
        """Only question and observed tool history enter the student's policy."""
        return {
            "ask_city": int(self.task.ask_city),
            "opened_project": int(self.opened_project),
            "opened_org": int(self.opened_org),
            "project_hit": int("project" in self.hits),
            "org_hit": int("org" in self.hits),
            "project_primary_empty": int(self.project_primary_empty),
            "org_primary_empty": int(self.org_primary_empty),
            "last_invalid": int(self.last_invalid),
        }

    def step(self, action: str) -> str:
        if self.done:
            raise RuntimeError("Cannot act after answer")
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        self.last_invalid = False
        if action == "answer":
            self.answer_given = (self.observed_city if self.task.ask_city else self.observed_organization) or "unknown"
            self.done = True
            observation = f"submitted: {self.answer_given}"
        else:
            self.tool_calls += 1
            observation = self._tool(action)

        self.history.append({"action": action, "observation": observation})
        return observation

    def _tool(self, action: str) -> str:
        if action == "search_project":
            if self.task.project_primary_miss:
                self.hits = ()
                self.project_primary_empty = True
            else:
                self.hits = ("project",)
            return f"search({self.task.project}): {list(self.hits)}"
        if action == "search_project_alias":
            self.hits = ("project",)
            return f"search(project alias {self.task.project}): {list(self.hits)}"
        if action == "open_project" and "project" in self.hits:
            self.opened_project = True
            self.observed_organization = self.task.organization
            self.hits = ()
            return f"Project {self.task.project} was developed by {self.observed_organization}."
        if action == "search_org" and self.opened_project:
            if self.task.org_primary_miss:
                self.hits = ()
                self.org_primary_empty = True
            else:
                self.hits = ("org",)
            return f"search({self.observed_organization}): {list(self.hits)}"
        if action == "search_org_alias" and self.opened_project:
            self.hits = ("org",)
            return f"search(organization alias {self.observed_organization}): {list(self.hits)}"
        if action == "open_org" and "org" in self.hits:
            self.opened_org = True
            self.observed_city = self.task.city
            self.hits = ()
            return f"{self.observed_organization} is based in {self.observed_city}."

        self.last_invalid = True
        self.invalid_calls += 1
        return "tool error: action requires a search hit or an opened project"


def teacher_action(world: World) -> str:
    """Reference policy. It chooses from observed state, never from the hidden answer."""
    if not world.opened_project:
        if "project" in world.hits:
            return "open_project"
        return "search_project_alias" if world.project_primary_empty else "search_project"
    if not world.task.ask_city:
        return "answer"
    if not world.opened_org:
        if "org" in world.hits:
            return "open_org"
        return "search_org_alias" if world.org_primary_empty else "search_org"
    return "answer"
