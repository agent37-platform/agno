"""Unit tests for the Agent37Tools toolkit.

The Agent37 REST API is replaced by an in-memory fake behind httpx.MockTransport,
so these tests run without network access or credentials.
"""

import base64
import json
import re
import subprocess
from functools import partial
from typing import Any, Dict, List
from unittest.mock import MagicMock

import httpx
import pytest

import agno.tools.agent37 as agent37_module
from agno.tools.agent37 import SESSION_STATE_SANDBOX_ID, Agent37Tools

TEST_API_KEY = "sk_live_test"

ALL_TOOLS = {
    "run_python_code",
    "run_command",
    "create_file",
    "read_file",
    "list_files",
    "delete_file",
    "get_public_url",
    "get_sandbox_info",
    "shutdown_sandbox",
}


class FakeAgent37:
    """Records every request and answers like the Agent37 API."""

    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []
        self.created = 0
        self.public_ports: List[Dict[str, Any]] = []
        self.exec_result: Dict[str, Any] = {"exit_code": 0, "stdout": "hello world", "stderr": "", "truncated": False}
        self.error: Any = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            return httpx.Response(self.error[0], json={"error": self.error[1]})
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        if request.method == "POST" and path == "/v1/instances":
            self.created += 1
            return httpx.Response(201, json=self._instance(f"sbx{self.created}"))
        if path.endswith("/exec"):
            return httpx.Response(200, json=self.exec_result)
        if path.endswith("/public-ports") and request.method == "GET":
            return httpx.Response(200, json={"data": self.public_ports})
        if path.endswith("/public-ports"):
            entry = {"port": body["port"], "url": f"https://slug{body['port']}.agent37.app"}
            self.public_ports.append(entry)
            return httpx.Response(201, json=entry)
        sandbox_id = path.rsplit("/", 1)[-1]
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": sandbox_id, "deleted": True})
        return httpx.Response(200, json=self._instance(sandbox_id))

    @staticmethod
    def _instance(sandbox_id: str) -> Dict[str, Any]:
        return {
            "id": sandbox_id,
            "status": "running",
            "template": "agent37-base",
            "type": "default",
            "region": "us",
            "resources": {"cpu": 2, "memory": 4, "disk": 4},
            "url": f"https://{sandbox_id}.agent37.app",
            "auto_sleep": True,
            "metadata": {"created_by": "agno"},
        }

    def calls(self, method: str, path_suffix: str) -> List[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path.endswith(path_suffix)]

    def exec_commands(self) -> List[str]:
        return [json.loads(r.content)["command"] for r in self.calls("POST", "/exec")]


@pytest.fixture
def api(monkeypatch) -> FakeAgent37:
    fake = FakeAgent37()
    transport = httpx.MockTransport(fake.handler)

    def request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.request(method, url, **kwargs)

    monkeypatch.setattr(agent37_module.httpx, "request", request)
    monkeypatch.setattr(agent37_module.httpx, "AsyncClient", partial(httpx.AsyncClient, transport=transport))
    return fake


@pytest.fixture
def agent() -> MagicMock:
    """An agent stub with a real dict session_state (so persistence logic works)."""
    a = MagicMock()
    a.session_state = {}
    return a


def _decoded_payload(command: str) -> str:
    return base64.b64decode(re.search(r"printf %s (\S*) \|", command).group(1)).decode()


# ---------------------------------------------------------------------------
# Initialization and registration
# ---------------------------------------------------------------------------
def test_init_with_api_key():
    assert Agent37Tools(api_key=TEST_API_KEY).api_key == TEST_API_KEY


def test_init_with_env_var(monkeypatch):
    monkeypatch.setenv("AGENT37_API_KEY", "sk_live_env")
    assert Agent37Tools().api_key == "sk_live_env"


def test_init_without_api_key_raises(monkeypatch):
    monkeypatch.delenv("AGENT37_API_KEY", raising=False)
    with pytest.raises(ValueError, match="AGENT37_API_KEY not set"):
        Agent37Tools()


def test_default_tools_registered_sync_and_async():
    tools = Agent37Tools(api_key=TEST_API_KEY)
    assert set(tools.functions.keys()) == ALL_TOOLS
    assert set(tools.async_functions.keys()) == ALL_TOOLS


def test_disable_individual_tool():
    tools = Agent37Tools(api_key=TEST_API_KEY, enable_shutdown_sandbox=False)
    assert "shutdown_sandbox" not in tools.functions
    assert "run_command" in tools.functions


def test_all_flag_overrides_disabled_tools():
    tools = Agent37Tools(api_key=TEST_API_KEY, enable_run_command=False, enable_read_file=False, all=True)
    assert set(tools.functions.keys()) == ALL_TOOLS


# ---------------------------------------------------------------------------
# Sandbox lifecycle
# ---------------------------------------------------------------------------
def test_first_call_creates_sandbox_and_stores_id(api, agent):
    result = Agent37Tools(api_key=TEST_API_KEY).run_command(agent, "echo hello")

    create = api.calls("POST", "/v1/instances")
    assert len(create) == 1
    assert create[0].url.host == "api.agent37.com"
    assert create[0].headers["Authorization"] == f"Bearer {TEST_API_KEY}"
    assert json.loads(create[0].content) == {"auto_sleep": True, "metadata": {"created_by": "agno"}}
    assert agent.session_state[SESSION_STATE_SANDBOX_ID] == "sbx1"

    exec_call = api.calls("POST", "/exec")[0]
    assert str(exec_call.url) == "https://sandbox.agent37.app/v1/instances/sbx1/exec"
    assert exec_call.headers["X-Agent37-Key"] == TEST_API_KEY
    assert json.loads(exec_call.content) == {"command": "echo hello"}
    assert "STDOUT:\nhello world" in result
    assert "Exit code: 0" in result


def test_sandbox_reused_across_calls(api, agent):
    tools = Agent37Tools(api_key=TEST_API_KEY)
    tools.run_command(agent, "echo one")
    tools.run_command(agent, "echo two")
    assert api.created == 1


def test_session_state_sandbox_reused_by_new_toolkit(api, agent):
    agent.session_state[SESSION_STATE_SANDBOX_ID] = "existing1"
    Agent37Tools(api_key=TEST_API_KEY).run_command(agent, "ls")
    assert api.created == 0
    assert api.calls("POST", "/exec")[0].url.path == "/v1/instances/existing1/exec"


def test_explicit_sandbox_id_never_creates(api, agent):
    Agent37Tools(api_key=TEST_API_KEY, sandbox_id="mine123").run_command(agent, "ls")
    assert api.created == 0
    assert api.calls("POST", "/exec")[0].url.path == "/v1/instances/mine123/exec"
    assert SESSION_STATE_SANDBOX_ID not in agent.session_state


def test_non_persistent_keeps_id_on_toolkit(api, agent):
    tools = Agent37Tools(api_key=TEST_API_KEY, persistent=False)
    tools.run_command(agent, "echo one")
    tools.run_command(agent, "echo two")
    assert api.created == 1
    assert SESSION_STATE_SANDBOX_ID not in agent.session_state


def test_template_and_create_params_sent_on_create(api, agent):
    tools = Agent37Tools(
        api_key=TEST_API_KEY,
        template="agent37-ubuntu",
        auto_sleep=False,
        create_params={"type": "vm", "region": "eu"},
    )
    tools.run_command(agent, "docker ps")
    assert json.loads(api.calls("POST", "/v1/instances")[0].content) == {
        "auto_sleep": False,
        "metadata": {"created_by": "agno"},
        "template": "agent37-ubuntu",
        "type": "vm",
        "region": "eu",
    }


def test_shutdown_deletes_and_forgets_sandbox(api, agent):
    tools = Agent37Tools(api_key=TEST_API_KEY)
    tools.run_command(agent, "ls")
    assert tools.shutdown_sandbox(agent) == "Sandbox sbx1 shut down."
    assert api.calls("DELETE", "/v1/instances/sbx1")
    assert SESSION_STATE_SANDBOX_ID not in agent.session_state

    tools.run_command(agent, "ls")
    assert api.created == 2


def test_shutdown_without_sandbox_does_not_create_one(api, agent):
    assert Agent37Tools(api_key=TEST_API_KEY).shutdown_sandbox(agent) == "No active sandbox to shut down."
    assert api.requests == []


# ---------------------------------------------------------------------------
# Tool behavior
# ---------------------------------------------------------------------------
def test_run_python_code_writes_script_and_runs_it(api, agent):
    Agent37Tools(api_key=TEST_API_KEY).run_python_code(agent, "print(true)")
    command = api.exec_commands()[0]
    assert _decoded_payload(command) == "print(True)"
    assert re.search(r"&& python3 /tmp/agno_run_\w+\.py$", command)


def test_create_file_command_round_trips_content(api, agent, tmp_path):
    content = "line1\n'single' \"double\" $HOME `backticks` \\ end\n"
    result = Agent37Tools(api_key=TEST_API_KEY).create_file(agent, "nested/dir/out.txt", content)
    assert result == "File written: nested/dir/out.txt"

    subprocess.run(["sh", "-c", api.exec_commands()[0]], cwd=tmp_path, check=True)
    assert (tmp_path / "nested/dir/out.txt").read_text() == content


def test_read_file_returns_stdout(api, agent):
    api.exec_result = {"exit_code": 0, "stdout": "file contents", "stderr": "", "truncated": False}
    assert Agent37Tools(api_key=TEST_API_KEY).read_file(agent, "my file.txt") == "file contents"
    assert api.exec_commands() == ["cat 'my file.txt'"]


def test_failed_command_in_file_tool_returns_error(api, agent):
    api.exec_result = {"exit_code": 1, "stdout": "", "stderr": "No such file", "truncated": False}
    result = json.loads(Agent37Tools(api_key=TEST_API_KEY).read_file(agent, "missing.txt"))
    assert result["status"] == "error"
    assert "No such file" in result["message"]


def test_run_command_reports_stderr_exit_code_and_truncation(api, agent):
    api.exec_result = {"exit_code": 2, "stdout": "", "stderr": "boom", "truncated": True}
    result = Agent37Tools(api_key=TEST_API_KEY).run_command(agent, "false")
    assert "STDERR:\nboom" in result
    assert "truncated" in result
    assert "Exit code: 2" in result


def test_list_and_delete_quote_paths(api, agent):
    tools = Agent37Tools(api_key=TEST_API_KEY)
    tools.list_files(agent, "my dir")
    tools.delete_file(agent, "a; rm -rf /")
    assert api.exec_commands() == ["ls -la 'my dir'", "rm -rf 'a; rm -rf /'"]


def test_get_public_url_creates_once_then_reuses(api, agent):
    tools = Agent37Tools(api_key=TEST_API_KEY)
    assert tools.get_public_url(agent, 8000) == "https://slug8000.agent37.app"
    assert tools.get_public_url(agent, 8000) == "https://slug8000.agent37.app"
    posts = api.calls("POST", "/public-ports")
    assert len(posts) == 1
    assert json.loads(posts[0].content) == {"port": 8000}


def test_get_sandbox_info_returns_summary(api, agent):
    info = json.loads(Agent37Tools(api_key=TEST_API_KEY).get_sandbox_info(agent))
    assert info["id"] == "sbx1"
    assert info["status"] == "running"
    assert "metadata" not in info


def test_api_error_is_returned_as_error_json(api, agent):
    api.error = (402, {"code": "insufficient_balance", "message": "Top up your wallet."})
    result = json.loads(Agent37Tools(api_key=TEST_API_KEY).run_command(agent, "ls"))
    assert result["status"] == "error"
    assert "402" in result["message"]
    assert "insufficient_balance: Top up your wallet." in result["message"]


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------
async def test_async_run_command_creates_and_execs(api, agent):
    result = await Agent37Tools(api_key=TEST_API_KEY).arun_command(agent, "echo hello")
    assert api.created == 1
    assert agent.session_state[SESSION_STATE_SANDBOX_ID] == "sbx1"
    assert api.calls("POST", "/exec")[0].headers["X-Agent37-Key"] == TEST_API_KEY
    assert "STDOUT:\nhello world" in result


async def test_async_file_tools(api, agent):
    tools = Agent37Tools(api_key=TEST_API_KEY)
    assert await tools.acreate_file(agent, "a.txt", "hi") == "File written: a.txt"
    api.exec_result = {"exit_code": 0, "stdout": "hi", "stderr": "", "truncated": False}
    assert await tools.aread_file(agent, "a.txt") == "hi"
    assert await tools.adelete_file(agent, "a.txt") == "Deleted: a.txt"
    assert _decoded_payload(api.exec_commands()[0]) == "hi"


async def test_async_public_url_info_and_shutdown(api, agent):
    tools = Agent37Tools(api_key=TEST_API_KEY)
    assert await tools.aget_public_url(agent, 3000) == "https://slug3000.agent37.app"
    assert json.loads(await tools.aget_sandbox_info(agent))["id"] == "sbx1"
    assert await tools.ashutdown_sandbox(agent) == "Sandbox sbx1 shut down."
    assert SESSION_STATE_SANDBOX_ID not in agent.session_state
