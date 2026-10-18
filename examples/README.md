# Examples

## `repo_assistant.py` — an agent that reads a project and asks before it writes

A small, complete agent in one commented file. Copy it into your project and replace the tools with yours.
It is tested offline in CI (`tests/test_examples.py`) and has opt-in live tests against a real model
(`pytest -m live`, `tests/live/test_examples_live.py`).

**1. Try it without an API key** (scripted answers, no network, no cost, leaves nothing behind):

```bash
python examples/repo_assistant.py --demo
```

You will see the agent list files, read the README, and then **stop and ask you** before it saves a note.
Answer `y` or `n`. Nothing is written until you approve.

**2. Use a real model** (Anthropic Claude; the cost of a question is a fraction of a cent with the default model):

```bash
pip install -e ".[anthropic]"          # or: uv sync --extra dev
export ANTHROPIC_API_KEY=...           # in your own shell. Never put the key in a file of this repository.
python examples/repo_assistant.py --root . --question "What does this project do? Save a short note."
python examples/repo_assistant.py      # interactive: ask several questions, empty line to quit
```

## What to look for

| You see | The harness guarantee behind it |
|---|---|
| The agent can read files but cannot read `.env`, `.git`, or keys, even if asked | `ReadFile` denies secrets by default; `list_files` hides them too |
| It stops before writing and shows the exact arguments | `save_note` is `Risk.WRITE`, the agent runs at `Autonomy.SUGGEST`: the run pauses (`waiting_for_approval`) |
| Your answer is tied to what you saw | `resume(..., pending_id=...)`: a stale approval can never apply to a different action (ADR 0014) |
| A bad note name (`../x.md`) is rejected without asking you | Arguments are validated before the approval step |
| The run ends after at most 10 steps, 60 000 tokens, or 120 s | `Limits` |
| `touch agent-notes/KILL` stops the next step | `KillSwitch` hook |
| Quit while it waits (Ctrl-C), then `--resume RUN_ID` continues (real mode; the run id and this hint are shown before you are asked; the demo keeps no runs) | Checkpoints (`.checkpoints/`, git-ignored) |
| `--verbose` shows what happened without any prompt or file content | Events carry metadata only (ADR 0011) |

The last line of every answer shows the run id, the status, the steps, and the tokens used (`in` excludes tokens
served from the prompt cache, which are listed separately; `tokens` is the total).
Exit codes: `0` finished, `1` the run ended otherwise (blocked by the kill file, failed, limit reached), `2` a problem with
the arguments or setup, `130` Ctrl-C. `--demo` keeps nothing (use `--notes-dir` / `--checkpoints-dir` to keep its files)
and cannot be combined with `--resume`.

## Safety notes

- The assistant reads **only** below `--root` and writes **only** into `--notes-dir` (default `./agent-notes`), one new
  file at a time, never overwriting. Every write needs your approval. `--yes` approves everything: use it only for demos and tests.
- Run it on a folder you are comfortable showing to the model provider: file contents you let it read are sent to the API.
- `.env` files are git-ignored in this repository and denied to the agent. Still, keep keys in your shell environment or a
  secrets manager, not in the project folder you point `--root` at.
- This is an example, not a hardened product. For real deployments read the production checklist in the main README.

## Adapting it

`build_agent()` is the part to change: the system prompt, the tools (`ListFiles`, `SaveNote` show how to write your own,
including input validation and a risk level), the autonomy level, the limits, and the checkpoint store. `run_question()`
and `resume_run()` show how to drive a run that pauses for a human; replace `input()` with a button in your own UI and send
back the `pending_id` that was shown with the request.
