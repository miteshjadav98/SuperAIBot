"""Publish changed code-default prompts to the prompt store.

Why this exists: ``core.prompts.get_prompt`` seeds a prompt into Mongo the first
time it is requested and *always serves the stored version afterwards*. That is
the right behaviour — an edit made in the Prompt Management UI must not be
silently reverted by a deploy — but it also means changing a default in code
reaches only environments that have never seen that prompt. Without this script,
a prompt change ships to new environments and nowhere else.

Usage (from ``backend/``)::

    python scripts/publish_prompts.py                 # dry run: what differs
    python scripts/publish_prompts.py --apply         # publish every difference
    python scripts/publish_prompts.py --apply --only superbot_supervisor_system
    python scripts/publish_prompts.py --diff          # show the full new text

Publishing is additive: each change becomes a new immutable version and the
registry pointer moves to it, so the previous version is one rollback away in
the Prompt Management UI. Nothing is ever overwritten or deleted.

**It will overwrite a hand-edit made in the UI**, because it cannot tell one
from a stale default — the old version survives, but the active pointer moves.
Read the dry run before passing --apply.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core import db  # noqa: E402
from core.prompts import active_content, declared_defaults, publish  # noqa: E402


def _load_declarations() -> dict[str, tuple[str, str]]:
    """Import everything that owns a prompt, so every default is declared.

    Agents declare theirs while building (the registry imports them); the Super
    Bot's are declared at module import by ``core.prompts.Prompt``.
    """
    import core.registry  # noqa: F401  — imports every agent module
    import superbot.executor  # noqa: F401
    import superbot.planner  # noqa: F401

    return declared_defaults()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the changes")
    parser.add_argument("--only", metavar="PROMPT_ID", help="restrict to one prompt")
    parser.add_argument("--diff", action="store_true", help="print the new content in full")
    args = parser.parse_args()

    if not db.mongo_configured():
        print("MONGODB_URI is not set — there is no prompt store to publish to.")
        return 1

    declared = _load_declarations()
    if args.only:
        declared = {k: v for k, v in declared.items() if k == args.only}
        if not declared:
            print(f"No prompt declared with id '{args.only}'.")
            return 1

    unchanged, new, changed = [], [], []
    for prompt_id, (name, default) in sorted(declared.items()):
        stored = active_content(prompt_id)
        if stored is None:
            new.append((prompt_id, name, default))
        elif stored.strip() == default.strip():
            unchanged.append(prompt_id)
        else:
            changed.append((prompt_id, name, default))

    print(f"{len(declared)} prompt(s) declared in code\n")
    print(f"  unchanged : {len(unchanged)}")
    for prompt_id in unchanged:
        print(f"      = {prompt_id}")
    print(f"  not stored: {len(new)}   (seeds itself on first use; publishing is optional)")
    for prompt_id, _, _ in new:
        print(f"      + {prompt_id}")
    print(f"  changed   : {len(changed)}")
    for prompt_id, _, _ in changed:
        print(f"      ~ {prompt_id}")

    if args.diff:
        for prompt_id, _, default in changed + new:
            print(f"\n--- {prompt_id} (new content) " + "-" * 30)
            print(default)

    targets = changed + new
    if not targets:
        print("\nNothing to publish.")
        return 0

    if not args.apply:
        print("\nDry run. Re-run with --apply to publish these.")
        return 0

    print()
    for prompt_id, name, default in targets:
        version = publish(prompt_id, default, name=name, description="Published from code default")
        print(f"  published {prompt_id} -> v{version}")
    print("\nRestart the agent service so it picks the new versions up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
