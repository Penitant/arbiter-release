"""Tests for the difficulty classifier: expected buckets fire, low-signal defaults to medium."""

from triage.classifier import classify


def _body(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_trivial_edit_is_easy():
    c = classify(_body("fix a typo"))
    assert c.difficulty == "easy"


def test_rename_is_easy():
    c = classify(_body("rename the variable foo to bar"))
    assert c.difficulty == "easy"


def test_hard_keywords_are_hard():
    c = classify(_body(
        "refactor the architecture to fix the race condition and optimize performance"
    ))
    assert c.difficulty == "hard"
    assert c.confidence > 0.5


def test_traceback_pushes_hard():
    c = classify(_body(
        "Traceback (most recent call last):\n  File 'a.py', line 3\nValueError: boom\n"
        "why does this keep happening when I run the test"
    ))
    assert c.difficulty == "hard"


def test_neutral_medium_length_request_is_medium():
    text = (
        "Please add a new REST endpoint that returns the current user's profile, "
        "including their display name and email, reading the values from the existing "
        "users table without changing any of the surrounding handlers."
    )
    c = classify(_body(text))
    assert c.difficulty == "medium"


def test_empty_or_unparseable_body_defaults_to_medium():
    assert classify(None).difficulty == "medium"
    assert classify({}).difficulty == "medium"
    assert classify({"messages": "not-a-list"}).difficulty == "medium"


def test_content_blocks_are_read():
    # Anthropic/OpenAI structured content blocks, not a plain string.
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "fix a typo"}]}]}
    assert classify(body).difficulty == "easy"
