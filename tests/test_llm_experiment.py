from agent_distillation_lab.llm_experiment import encode_supervised, messages, parse_action, successful_steps
from agent_distillation_lab.world import Task, World


def test_prompt_contains_observations_but_not_hidden_answer():
    world = World(Task("P1", "Org1", "Lima", True))
    prompt = messages(world)
    assert "P1" in prompt[1]["content"]
    assert "Org1" not in prompt[1]["content"]
    assert "Lima" not in prompt[1]["content"]
    world.step("search_project")
    world.step("open_project")
    assert "Org1" in messages(world)[1]["content"]
    assert "Lima" not in messages(world)[1]["content"]


def test_action_parser_rejects_unstructured_answers():
    assert parse_action('{"action":"search_project"}') == "search_project"
    assert parse_action("search_project") == "search_project"
    assert parse_action("Action: open_org") == "open_org"
    assert parse_action("I think we should answer") is None


def test_only_successful_teacher_traces_enter_training(tmp_path):
    data = tmp_path / "teacher.jsonl"
    data.write_text(
        '{"success": true, "steps": [{"action": "search_project"}]}\n'
        '{"success": false, "steps": [{"action": "answer"}]}\n',
        encoding="utf-8",
    )
    rows, successes = successful_steps(data)
    assert successes == 1
    assert rows == [{"action": "search_project"}]


def test_training_loss_excludes_prompt_and_tool_observations():
    class Tokenizer:
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            assert any("Observation:" in message["content"] for message in messages)
            return [10, 11, 12]

        def __call__(self, text, **kwargs):
            assert text == "open_project<eos>"
            return {"input_ids": [20, 21]}

    world = World(Task("P1", "Org1", "Lima", True))
    world.step("search_project")
    encoded = encode_supervised(Tokenizer(), messages(world), "open_project")
    assert encoded["labels"] == [-100, -100, -100, 20, 21]
