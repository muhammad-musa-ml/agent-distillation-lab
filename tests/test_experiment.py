from agent_distillation_lab.experiment import demonstrations, first_error_corrections, rollout, run
from agent_distillation_lab.policy import Student
from agent_distillation_lab.supervision import char_loss_mask, teacher_segments
from agent_distillation_lab.world import Task, World, make_tasks, teacher_action


def test_teacher_recovers_from_both_empty_searches():
    task = Task("P1", "Lab1", "Lima", True, True, True)
    world = rollout(task, teacher_action)
    assert world.answer_given == "Lima"
    assert [item["action"] for item in world.history] == [
        "search_project", "search_project_alias", "open_project",
        "search_org", "search_org_alias", "open_org", "answer",
    ]


def test_observation_is_input_and_never_target():
    task = Task("P2", "Lab2", "Kyoto", True)
    rows = demonstrations([task])
    assert [row.action for row in rows] == [
        "search_project", "open_project", "search_org", "open_org", "answer"
    ]
    world = World(task)
    before = world.features()
    world.step("search_project")
    assert world.features() != before
    assert all("Kyoto" not in row.action for row in rows)
    segments = teacher_segments(task)
    text, mask = char_loss_mask(segments)
    assert len(text) == len(mask)
    assert all(not train for segment in segments if segment["role"] == "tool" for train in [segment["train"]])
    assert any(mask)
    assert all(mask[i] == 0 for i in range(text.index("Observation:"), text.index("Observation:") + len("Observation:")))


def test_correction_data_starts_at_student_error():
    student = Student()
    student.fit(demonstrations(make_tasks(40, 11)))
    rows, count = first_error_corrections(make_tasks(40, 23, 1.0), student)
    assert count > 0
    assert rows
    assert rows[0].action == "search_project_alias"


def test_corpus_change_changes_answer_without_retraining():
    student = Student()
    student.fit(demonstrations(make_tasks(80, 11)))
    task_a = Task("P3", "Lab3", "Oslo", True)
    task_b = Task("P3", "Lab3", "Tunis", True)
    assert rollout(task_a, student).answer_given == "Oslo"
    assert rollout(task_b, student).answer_given == "Tunis"


def test_repair_improves_failure_split():
    report = run(120, 100, 100)
    assert report["search_failures"]["corrected"]["success_rate"] > report["search_failures"]["clone"]["success_rate"]
