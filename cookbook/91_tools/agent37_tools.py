"""
Agent with Agent37 tools

This example shows how to use Agno's Agent37 integration to run agent-generated
code in a persistent cloud sandbox.

1. Get your Agent37 API key: https://www.agent37.com/dashboard/cloud/api-keys
2. Set the API key as an environment variable:
    export AGENT37_API_KEY=sk_live_...
3. Install the dependencies:
    uv pip install agno openai

The sandbox keeps its files and installed packages across tool calls, and across
runs when persistent=True. It sleeps when idle and wakes on the next tool call.
"""

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.agent37 import Agent37Tools

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

# Every tool has its own enable_* flag, so you can turn tools off individually:
#   Agent37Tools(enable_shutdown_sandbox=False)
# Sandboxes default to agent37-base (Python, Node, git). Pick another template or shape:
#   Agent37Tools(template="agent37-ubuntu", create_params={"type": "vm"})  # Docker inside
#   Agent37Tools(create_params={"resources": {"cpu": 4, "memory": 8}})
# Reuse a sandbox you already have:
#   Agent37Tools(sandbox_id="ab12cd34ef")

agent = Agent(
    name="Coding Agent with Agent37 tools",
    model=OpenAIResponses(id="gpt-5.5"),
    tools=[Agent37Tools()],
    markdown=True,
    instructions=[
        "You are an expert at writing and executing code in an Agent37 sandbox.",
        "Your primary purpose is to:",
        "1. Write clear, efficient code based on user requests",
        "2. ALWAYS execute the code in the sandbox using run_python_code or run_command",
        "3. Show the actual execution results to the user",
        "4. Provide explanations of how the code works and what the output means",
        "Guidelines:",
        "- NEVER just provide code without executing it",
        "- Install missing packages when needed using run_command, for example pip install <package>",
        "- Use file operations (create_file, read_file, list_files) when working with scripts",
        "- To share a web app, start it in the background and call get_public_url with its port",
    ],
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent.print_response(
        "Install sympy, compute the 10th Fibonacci number, and tell me whether it is prime"
    )
