"""Test-wide setup: force hermetic config flags before the app is imported."""

import os

# No startup discovery network calls.
os.environ["ARBITER_DISCOVERY"] = "0"
# Don't read the developer's real OpenCode auth.json.
os.environ["ARBITER_READ_OPENCODE_AUTH"] = "0"
# Don't enqueue fine-tune gradient steps when tests post to /corrections.
os.environ["ARBITER_FINETUNE"] = "0"
