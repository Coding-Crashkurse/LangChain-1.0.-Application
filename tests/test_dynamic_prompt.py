from app.chat.dynamic_prompting import preview_prompt


def test_beginner_prompt_simple():
    prompt = preview_prompt("beginner")
    assert "very simple language" in prompt.lower()


def test_mid_prompt_mentions_code():
    prompt = preview_prompt("mid")
    assert "code" in prompt.lower()


def test_expert_prompt_is_technical():
    prompt = preview_prompt("expert", feedback="go deeper")
    assert "deeply technical" in prompt.lower()
    assert "go deeper" in prompt.lower()
