from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain.agents.middleware import wrap_model_call, dynamic_prompt, HumanInTheLoopMiddleware
from langchain.agents.middleware import ModelRequest, ModelResponse
from typing import Callable, Awaitable

load_dotenv()

from llm.factory import get_chat_model

azure_model = get_chat_model()


@dataclass
class EmailContext:
    email_address: str = "julie@example.com"
    password: str = "password123"


class AuthenticatedState(AgentState):
    authenticated: bool


@tool
def check_inbox() -> str:
    """Check the inbox for recent emails"""
    return """
    Hi Julie, 
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an response email"""
    return f"Email sent to {to} with subject {subject} and body {body}"


@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the user with the given email and password"""
    if email == runtime.context.email_address and password == runtime.context.password:
        return Command(
            update={
                "authenticated": True,
                "messages": [
                    ToolMessage("Successfully authenticated", tool_call_id=runtime.tool_call_id)
                ],
            }
        )
    else:
        return Command(
            update={
                "authenticated": False,
                "messages": [
                    ToolMessage("Authentication failed", tool_call_id=runtime.tool_call_id)
                ],
            }
        )


@wrap_model_call
async def dynamic_tool_call(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """Allow read inbox and send email tools only if user provides correct email and password"""

    authenticated = request.state.get("authenticated")

    if authenticated:
        tools = [check_inbox, send_email]
    else:
        tools = [authenticate]

    request = request.override(tools=tools)
    return await handler(request)


from core.prompts import get_prompt

authenticated_prompt = get_prompt(
    "email_authenticated_system",
    "You are a helpful assistant that can check the inbox and send emails.",
    name="Email Agent — Authenticated Prompt",
)
unauthenticated_prompt = get_prompt(
    "email_unauthenticated_system",
    "You are a helpful assistant that can authenticate users.",
    name="Email Agent — Unauthenticated Prompt",
)


@dynamic_prompt
def dynamic_prompt_func(request: ModelRequest) -> str:
    """Generate system prompt based on authentication status"""
    authenticated = request.state.get("authenticated")

    if authenticated:
        return authenticated_prompt
    else:
        return unauthenticated_prompt


agent = create_agent(
        azure_model,
        tools=[authenticate, check_inbox, send_email],
        state_schema=AuthenticatedState,
        context_schema=EmailContext,
        middleware=[
            dynamic_tool_call,
            dynamic_prompt_func,
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "authenticate": False,
                    "check_inbox": False,
                    "send_email": True,
                }
            ),
        ],
    )

from core.base_agent import AgentManifest

MANIFEST = AgentManifest(
    id="email_agent",
    label="Email Agent",
    emoji="✉️",
    description=(
        "Authenticates a user, reads their inbox, drafts replies, and sends "
        "email — with human approval required before anything is sent. Use for "
        "anything about email: reading, drafting, writing to someone, or sending mail."
    ),
    agent_type="langchain",
    builder=lambda: agent,
    capabilities=["email"],
)
