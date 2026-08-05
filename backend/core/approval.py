"""Declarative human-in-the-loop approval for destructive tools.

The problem this solves: approval used to be a hand-maintained map inside each
agent —

    HumanInTheLoopMiddleware(interrupt_on={
        "authenticate": False, "check_inbox": False, "send_email": True,
    })

which has to be edited every time a tool is added, and silently fails open when
someone forgets. The safety property lived in the agent's wiring rather than
with the dangerous thing itself.

Here, a tool declares its own risk at the point of definition::

    @requires_approval(describe=lambda args: f"Send email to {args['to']}")
    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        ...

and the agent asks for a policy over whatever tools it happens to have::

    middleware=[approval_middleware([send_email, search_emails, ...])]

Add a destructive tool later and it is gated automatically — there is no second
place to remember to update. Nothing is marked by default, so the failure mode
of forgetting to *unmark* something is an extra prompt, not an unwanted send.

Built on LangChain's ``HumanInTheLoopMiddleware`` rather than replacing it: the
interrupt/resume mechanics, decision types and payload shape are maintained
upstream. This module only decides *which* tools get gated and how the request
is described to the human.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain_core.tools import BaseTool

# tool name -> the policy to apply. Module-level because tools are defined once
# at import time, and a tool's risk is a property of the tool, not of the run.
_POLICY: dict[str, InterruptOnConfig] = {}

Describe = Callable[[dict[str, Any]], str]


def requires_approval(
    tool: BaseTool | None = None,
    *,
    describe: Describe | None = None,
    allow_edit: bool = True,
):
    """Mark a tool as requiring human approval before it executes.

    Args:
        tool: The tool, when used as a bare decorator.
        describe: Builds the message shown to the human from the tool call's
            arguments. Worth providing — "Send email to jane@example.com,
            subject 'Coffee?'" is reviewable; a raw JSON blob is not.
        allow_edit: Whether the human may amend the arguments before approving.
            Leave on for anything a human would plausibly want to reword;
            turn it off where edited arguments could not be validated.

    Returns:
        The same tool, unchanged. Only the policy registry is updated, so a
        marked tool stays usable anywhere — including in tests, where no
        approval machinery is involved.
    """

    def mark(target: BaseTool) -> BaseTool:
        decisions = ["approve", "edit", "reject"] if allow_edit else ["approve", "reject"]
        config: InterruptOnConfig = {"allowed_decisions": decisions}

        if describe is not None:
            # HumanInTheLoopMiddleware passes (tool_call, state, runtime); the
            # arguments are the only part a reviewer needs, so `describe` takes
            # just those and stays trivial to write and to unit-test.
            config["description"] = lambda tool_call, *_: describe(tool_call["args"])

        _POLICY[target.name] = config
        return target

    return mark(tool) if tool is not None else mark


def is_gated(tool_name: str) -> bool:
    """Whether ``tool_name`` has been marked as requiring approval."""
    return tool_name in _POLICY


def approval_middleware(
    tools: Iterable[BaseTool],
    *,
    overrides: dict[str, bool | InterruptOnConfig] | None = None,
) -> HumanInTheLoopMiddleware:
    """Build the approval middleware for ``tools`` from their declared policies.

    Tools with no policy are auto-approved (that is the middleware's own
    default, so unmarked tools cost nothing). ``overrides`` is the escape hatch
    for the rare case where one agent needs to gate — or ungate — a shared tool.
    """
    interrupt_on: dict[str, bool | InterruptOnConfig] = {
        tool.name: _POLICY[tool.name] for tool in tools if tool.name in _POLICY
    }
    interrupt_on.update(overrides or {})
    return HumanInTheLoopMiddleware(interrupt_on=interrupt_on)
