import base64
import json
import posixpath
import shlex
from os import getenv
from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import httpx

from agno.agent import Agent
from agno.team import Team
from agno.tools import Toolkit
from agno.utils.code_execution import prepare_python_code
from agno.utils.log import log_info

API_URL = "https://api.agent37.com/v1"
# Commands are sent straight to the host the sandbox runs on, which skips a hop through the API.
EXEC_URL = "https://sandbox.agent37.app/v1"

DEFAULT_INSTRUCTIONS = dedent(
    """\
    You have access to an Agent37 sandbox: a persistent Linux machine with Python, Node, and git.
    Files you write and packages you install stay available across tool calls.
    Available tools:
    - `run_python_code`: Execute Python code and return its output
    - `run_command`: Execute a shell command
    - `create_file`: Create or overwrite a file
    - `read_file`: Read a file's contents
    - `list_files`: List the contents of a directory
    - `delete_file`: Delete a file or directory
    - `get_public_url`: Get a public HTTPS URL for a port a process in the sandbox listens on
    - `get_sandbox_info`: Inspect the current sandbox
    - `shutdown_sandbox`: Delete the current sandbox
    When asked to run or verify code, write it, execute it with run_python_code or run_command, and show the real output.
    Install missing packages with run_command, for example `pip install <package>`.
    Start servers and other long-running processes in the background with their output sent to a file,
    for example `nohup python3 -m http.server 8000 > server.log 2>&1 &`.
    """
)

# Key under which the sandbox id is stored in the agent's session state, so the
# same sandbox is reused across tool calls and across runs.
SESSION_STATE_SANDBOX_ID = "agent37_sandbox_id"


class Agent37Tools(Toolkit):
    """Run agent-generated code in a persistent Agent37 sandbox.

    The toolkit talks to the Agent37 REST API directly, so it needs no extra
    package. Every tool has both a sync and an async variant, so the toolkit
    works with `agent.run()` and `agent.arun()`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        sandbox_id: Optional[str] = None,
        template: Optional[str] = None,
        auto_sleep: bool = True,
        create_params: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
        persistent: bool = True,
        enable_run_python_code: bool = True,
        enable_run_command: bool = True,
        enable_create_file: bool = True,
        enable_read_file: bool = True,
        enable_list_files: bool = True,
        enable_delete_file: bool = True,
        enable_get_public_url: bool = True,
        enable_get_sandbox_info: bool = True,
        enable_shutdown_sandbox: bool = True,
        all: bool = False,
        instructions: Optional[str] = None,
        add_instructions: bool = False,
        **kwargs: Any,
    ):
        """Initialize the Agent37 toolkit.

        Args:
            api_key: Agent37 API key (defaults to the AGENT37_API_KEY env var).
            sandbox_id: Use an existing sandbox instead of creating a new one.
            template: Template to create the sandbox from (defaults to `agent37-base`,
                a Linux sandbox with Python, Node, and git).
            auto_sleep: Let the sandbox sleep when idle, billing disk alone until the
                next tool call wakes it (default: True).
            create_params: Extra fields for the create request, for example
                {"type": "vm"} or {"resources": {"cpu": 4, "memory": 8}}.
            timeout: Seconds to wait for a command to finish (default: 300). A command
                that takes longer keeps running in the sandbox.
            persistent: Store the sandbox id in the agent's session state so the same
                sandbox is reused across runs (default: True).
            enable_run_python_code: Register the run_python_code tool (default: True).
            enable_run_command: Register the run_command tool (default: True).
            enable_create_file: Register the create_file tool (default: True).
            enable_read_file: Register the read_file tool (default: True).
            enable_list_files: Register the list_files tool (default: True).
            enable_delete_file: Register the delete_file tool (default: True).
            enable_get_public_url: Register the get_public_url tool (default: True).
            enable_get_sandbox_info: Register the get_sandbox_info tool (default: True).
            enable_shutdown_sandbox: Register the shutdown_sandbox tool (default: True).
            all: Register every tool, overriding the individual enable_* flags (default: False).
            instructions: Override the default toolkit instructions.
            add_instructions: Whether to add the instructions to the agent's system message.
        """
        self.api_key = api_key or getenv("AGENT37_API_KEY")
        if not self.api_key:
            raise ValueError("AGENT37_API_KEY not set. Please set the AGENT37_API_KEY environment variable.")

        self.sandbox_id = sandbox_id
        self.template = template
        self.auto_sleep = auto_sleep
        self.create_params = create_params or {}
        self.timeout = timeout
        self.persistent = persistent
        # The sandbox created by this toolkit when persistent=False.
        self._created_sandbox_id: Optional[str] = None

        self.instructions = instructions or DEFAULT_INSTRUCTIONS

        tools: List[Any] = []
        async_tools: List[Any] = []
        if all or enable_run_python_code:
            tools.append(self.run_python_code)
            async_tools.append((self.arun_python_code, "run_python_code"))
        if all or enable_run_command:
            tools.append(self.run_command)
            async_tools.append((self.arun_command, "run_command"))
        if all or enable_create_file:
            tools.append(self.create_file)
            async_tools.append((self.acreate_file, "create_file"))
        if all or enable_read_file:
            tools.append(self.read_file)
            async_tools.append((self.aread_file, "read_file"))
        if all or enable_list_files:
            tools.append(self.list_files)
            async_tools.append((self.alist_files, "list_files"))
        if all or enable_delete_file:
            tools.append(self.delete_file)
            async_tools.append((self.adelete_file, "delete_file"))
        if all or enable_get_public_url:
            tools.append(self.get_public_url)
            async_tools.append((self.aget_public_url, "get_public_url"))
        if all or enable_get_sandbox_info:
            tools.append(self.get_sandbox_info)
            async_tools.append((self.aget_sandbox_info, "get_sandbox_info"))
        if all or enable_shutdown_sandbox:
            tools.append(self.shutdown_sandbox)
            async_tools.append((self.ashutdown_sandbox, "shutdown_sandbox"))

        super().__init__(
            name="agent37_tools",
            tools=tools,
            async_tools=async_tools,
            instructions=self.instructions,
            add_instructions=add_instructions,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _headers(self, url: str) -> Dict[str, str]:
        if url.startswith(EXEC_URL):
            return {"X-Agent37-Key": str(self.api_key)}
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _parse(response: httpx.Response) -> Any:
        if response.is_error:
            try:
                error = response.json()["error"]
                message = f"{error['code']}: {error['message']}"
            except Exception:
                message = response.text
            raise RuntimeError(f"Agent37 API returned {response.status_code}: {message}")
        return response.json()

    def _request(self, method: str, url: str, body: Optional[Dict[str, Any]] = None, timeout: float = 120) -> Any:
        response = httpx.request(method, url, headers=self._headers(url), json=body, timeout=timeout)
        return self._parse(response)

    async def _arequest(
        self, method: str, url: str, body: Optional[Dict[str, Any]] = None, timeout: float = 120
    ) -> Any:
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, headers=self._headers(url), json=body, timeout=timeout)
        return self._parse(response)

    # ------------------------------------------------------------------
    # Sandbox lifecycle helpers
    # ------------------------------------------------------------------
    def _current_sandbox_id(self, agent: Optional[Union[Agent, Team]]) -> Optional[str]:
        if self.sandbox_id:
            return self.sandbox_id
        if self.persistent and agent is not None and hasattr(agent, "session_state"):
            if agent.session_state is None:
                agent.session_state = {}
            stored = agent.session_state.get(SESSION_STATE_SANDBOX_ID)
            return stored if isinstance(stored, str) else None
        return self._created_sandbox_id

    def _remember_sandbox(self, agent: Optional[Union[Agent, Team]], sandbox_id: str) -> None:
        if self.persistent and agent is not None and hasattr(agent, "session_state"):
            if agent.session_state is None:
                agent.session_state = {}
            agent.session_state[SESSION_STATE_SANDBOX_ID] = sandbox_id
        else:
            self._created_sandbox_id = sandbox_id
        log_info(f"Created Agent37 sandbox: {sandbox_id}")

    def _forget_sandbox(self, agent: Optional[Union[Agent, Team]]) -> None:
        self.sandbox_id = None
        self._created_sandbox_id = None
        if agent is not None and hasattr(agent, "session_state") and agent.session_state:
            agent.session_state.pop(SESSION_STATE_SANDBOX_ID, None)

    def _create_body(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"auto_sleep": self.auto_sleep, "metadata": {"created_by": "agno"}}
        if self.template:
            body["template"] = self.template
        body.update(self.create_params)
        return body

    def _get_sandbox_id(self, agent: Optional[Union[Agent, Team]]) -> str:
        sandbox_id = self._current_sandbox_id(agent)
        if sandbox_id:
            return sandbox_id
        sandbox = self._request("POST", f"{API_URL}/instances", self._create_body(), timeout=300)
        self._remember_sandbox(agent, sandbox["id"])
        return sandbox["id"]

    async def _aget_sandbox_id(self, agent: Optional[Union[Agent, Team]]) -> str:
        sandbox_id = self._current_sandbox_id(agent)
        if sandbox_id:
            return sandbox_id
        sandbox = await self._arequest("POST", f"{API_URL}/instances", self._create_body(), timeout=300)
        self._remember_sandbox(agent, sandbox["id"])
        return sandbox["id"]

    def _exec(self, agent: Union[Agent, Team], command: str) -> Dict[str, Any]:
        sandbox_id = self._get_sandbox_id(agent)
        return self._request("POST", f"{EXEC_URL}/instances/{sandbox_id}/exec", {"command": command}, self.timeout)

    async def _aexec(self, agent: Union[Agent, Team], command: str) -> Dict[str, Any]:
        sandbox_id = await self._aget_sandbox_id(agent)
        return await self._arequest(
            "POST", f"{EXEC_URL}/instances/{sandbox_id}/exec", {"command": command}, self.timeout
        )

    # ------------------------------------------------------------------
    # Command and formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _write_command(file_path: str, content: str) -> str:
        encoded = base64.b64encode(content.encode()).decode()
        parent = posixpath.dirname(file_path) or "."
        return f"mkdir -p {shlex.quote(parent)} && printf %s {encoded} | base64 -d > {shlex.quote(file_path)}"

    @classmethod
    def _python_command(cls, code: str) -> str:
        path = f"/tmp/agno_run_{uuid4().hex[:8]}.py"
        return f"{cls._write_command(path, prepare_python_code(code))} && python3 {path}"

    @staticmethod
    def _error(message: str, error: Union[str, Exception]) -> str:
        return json.dumps({"status": "error", "message": f"{message}: {str(error)}"})

    @staticmethod
    def _format_result(result: Dict[str, Any]) -> str:
        parts: List[str] = []
        if result.get("stdout"):
            parts.append(f"STDOUT:\n{result['stdout']}")
        if result.get("stderr"):
            parts.append(f"STDERR:\n{result['stderr']}")
        if result.get("truncated"):
            parts.append("Output was truncated; send large output to a file and read parts of it.")
        parts.append(f"Exit code: {result['exit_code']}")
        return "\n".join(parts)

    @staticmethod
    def _sandbox_summary(sandbox: Dict[str, Any]) -> str:
        keys = ["id", "status", "template", "type", "region", "resources", "url", "auto_sleep"]
        return json.dumps({key: sandbox.get(key) for key in keys})

    @staticmethod
    def _find_public_url(ports: Dict[str, Any], port: int) -> Optional[str]:
        for entry in ports.get("data", []):
            if entry.get("port") == port:
                return entry.get("url")
        return None

    # ------------------------------------------------------------------
    # Tools (sync)
    # ------------------------------------------------------------------
    def run_python_code(self, agent: Union[Agent, Team], code: str) -> str:
        """Execute Python code in the sandbox and return its output.

        Args:
            code: Python code to execute.

        Returns:
            The output (stdout, stderr, exit code) or an error message.
        """
        try:
            return self._format_result(self._exec(agent, self._python_command(code)))
        except Exception as e:
            return self._error("Error executing code", e)

    def run_command(self, agent: Union[Agent, Team], command: str) -> str:
        """Execute a shell command in the sandbox. Commands start in the home directory.

        Args:
            command: Shell command to execute.

        Returns:
            The output (stdout, stderr, exit code) or an error message.
        """
        try:
            return self._format_result(self._exec(agent, command))
        except Exception as e:
            return self._error("Error executing command", e)

    def create_file(self, agent: Union[Agent, Team], file_path: str, content: str) -> str:
        """Create or overwrite a file in the sandbox, creating parent directories as needed.

        Args:
            file_path: Path to the file, absolute or relative to the home directory.
            content: Text content to write.

        Returns:
            A success message or an error message.
        """
        try:
            result = self._exec(agent, self._write_command(file_path, content))
            if result["exit_code"] != 0:
                return self._error(f"Error writing {file_path}", result["stderr"])
            return f"File written: {file_path}"
        except Exception as e:
            return self._error("Error creating file", e)

    def read_file(self, agent: Union[Agent, Team], file_path: str) -> str:
        """Read a text file from the sandbox.

        Args:
            file_path: Path to the file, absolute or relative to the home directory.

        Returns:
            The file contents or an error message.
        """
        try:
            result = self._exec(agent, f"cat {shlex.quote(file_path)}")
            if result["exit_code"] != 0:
                return self._error(f"Error reading {file_path}", result["stderr"])
            return result["stdout"]
        except Exception as e:
            return self._error("Error reading file", e)

    def list_files(self, agent: Union[Agent, Team], directory: str = ".") -> str:
        """List the contents of a directory in the sandbox.

        Args:
            directory: Directory to list, absolute or relative to the home directory (default: home).

        Returns:
            The directory listing or an error message.
        """
        try:
            result = self._exec(agent, f"ls -la {shlex.quote(directory)}")
            if result["exit_code"] != 0:
                return self._error(f"Error listing {directory}", result["stderr"])
            return f"Contents of {directory}:\n{result['stdout']}"
        except Exception as e:
            return self._error("Error listing files", e)

    def delete_file(self, agent: Union[Agent, Team], file_path: str) -> str:
        """Delete a file or directory in the sandbox.

        Args:
            file_path: Path to the file or directory, absolute or relative to the home directory.

        Returns:
            A success message or an error message.
        """
        try:
            result = self._exec(agent, f"rm -rf {shlex.quote(file_path)}")
            if result["exit_code"] != 0:
                return self._error(f"Error deleting {file_path}", result["stderr"])
            return f"Deleted: {file_path}"
        except Exception as e:
            return self._error("Error deleting file", e)

    def get_public_url(self, agent: Union[Agent, Team], port: int) -> str:
        """Get a public HTTPS URL for a port a process in the sandbox listens on.

        Anyone with the URL can reach the port, so put authentication in front of anything sensitive.

        Args:
            port: Port a process inside the sandbox is listening on (bind it to 0.0.0.0).

        Returns:
            The public URL or an error message.
        """
        try:
            sandbox_id = self._get_sandbox_id(agent)
            ports_url = f"{API_URL}/instances/{sandbox_id}/public-ports"
            url = self._find_public_url(self._request("GET", ports_url), port)
            if url is None:
                url = self._request("POST", ports_url, {"port": port})["url"]
            return str(url)
        except Exception as e:
            return self._error("Error getting public URL", e)

    def get_sandbox_info(self, agent: Union[Agent, Team]) -> str:
        """Get information about the current sandbox.

        Returns:
            JSON with the sandbox id, status, template, shape, and URL, or an error message.
        """
        try:
            sandbox_id = self._get_sandbox_id(agent)
            return self._sandbox_summary(self._request("GET", f"{API_URL}/instances/{sandbox_id}"))
        except Exception as e:
            return self._error("Error getting sandbox info", e)

    def shutdown_sandbox(self, agent: Union[Agent, Team]) -> str:
        """Delete the current sandbox and everything in it.

        Returns:
            A success message or an error message.
        """
        try:
            sandbox_id = self._current_sandbox_id(agent)
            if not sandbox_id:
                return "No active sandbox to shut down."
            self._request("DELETE", f"{API_URL}/instances/{sandbox_id}")
            self._forget_sandbox(agent)
            return f"Sandbox {sandbox_id} shut down."
        except Exception as e:
            return self._error("Error shutting down sandbox", e)

    # ------------------------------------------------------------------
    # Tools (async)
    # ------------------------------------------------------------------
    async def arun_python_code(self, agent: Union[Agent, Team], code: str) -> str:
        """Execute Python code in the sandbox and return its output.

        Args:
            code: Python code to execute.

        Returns:
            The output (stdout, stderr, exit code) or an error message.
        """
        try:
            return self._format_result(await self._aexec(agent, self._python_command(code)))
        except Exception as e:
            return self._error("Error executing code", e)

    async def arun_command(self, agent: Union[Agent, Team], command: str) -> str:
        """Execute a shell command in the sandbox. Commands start in the home directory.

        Args:
            command: Shell command to execute.

        Returns:
            The output (stdout, stderr, exit code) or an error message.
        """
        try:
            return self._format_result(await self._aexec(agent, command))
        except Exception as e:
            return self._error("Error executing command", e)

    async def acreate_file(self, agent: Union[Agent, Team], file_path: str, content: str) -> str:
        """Create or overwrite a file in the sandbox, creating parent directories as needed.

        Args:
            file_path: Path to the file, absolute or relative to the home directory.
            content: Text content to write.

        Returns:
            A success message or an error message.
        """
        try:
            result = await self._aexec(agent, self._write_command(file_path, content))
            if result["exit_code"] != 0:
                return self._error(f"Error writing {file_path}", result["stderr"])
            return f"File written: {file_path}"
        except Exception as e:
            return self._error("Error creating file", e)

    async def aread_file(self, agent: Union[Agent, Team], file_path: str) -> str:
        """Read a text file from the sandbox.

        Args:
            file_path: Path to the file, absolute or relative to the home directory.

        Returns:
            The file contents or an error message.
        """
        try:
            result = await self._aexec(agent, f"cat {shlex.quote(file_path)}")
            if result["exit_code"] != 0:
                return self._error(f"Error reading {file_path}", result["stderr"])
            return result["stdout"]
        except Exception as e:
            return self._error("Error reading file", e)

    async def alist_files(self, agent: Union[Agent, Team], directory: str = ".") -> str:
        """List the contents of a directory in the sandbox.

        Args:
            directory: Directory to list, absolute or relative to the home directory (default: home).

        Returns:
            The directory listing or an error message.
        """
        try:
            result = await self._aexec(agent, f"ls -la {shlex.quote(directory)}")
            if result["exit_code"] != 0:
                return self._error(f"Error listing {directory}", result["stderr"])
            return f"Contents of {directory}:\n{result['stdout']}"
        except Exception as e:
            return self._error("Error listing files", e)

    async def adelete_file(self, agent: Union[Agent, Team], file_path: str) -> str:
        """Delete a file or directory in the sandbox.

        Args:
            file_path: Path to the file or directory, absolute or relative to the home directory.

        Returns:
            A success message or an error message.
        """
        try:
            result = await self._aexec(agent, f"rm -rf {shlex.quote(file_path)}")
            if result["exit_code"] != 0:
                return self._error(f"Error deleting {file_path}", result["stderr"])
            return f"Deleted: {file_path}"
        except Exception as e:
            return self._error("Error deleting file", e)

    async def aget_public_url(self, agent: Union[Agent, Team], port: int) -> str:
        """Get a public HTTPS URL for a port a process in the sandbox listens on.

        Anyone with the URL can reach the port, so put authentication in front of anything sensitive.

        Args:
            port: Port a process inside the sandbox is listening on (bind it to 0.0.0.0).

        Returns:
            The public URL or an error message.
        """
        try:
            sandbox_id = await self._aget_sandbox_id(agent)
            ports_url = f"{API_URL}/instances/{sandbox_id}/public-ports"
            url = self._find_public_url(await self._arequest("GET", ports_url), port)
            if url is None:
                url = (await self._arequest("POST", ports_url, {"port": port}))["url"]
            return str(url)
        except Exception as e:
            return self._error("Error getting public URL", e)

    async def aget_sandbox_info(self, agent: Union[Agent, Team]) -> str:
        """Get information about the current sandbox.

        Returns:
            JSON with the sandbox id, status, template, shape, and URL, or an error message.
        """
        try:
            sandbox_id = await self._aget_sandbox_id(agent)
            return self._sandbox_summary(await self._arequest("GET", f"{API_URL}/instances/{sandbox_id}"))
        except Exception as e:
            return self._error("Error getting sandbox info", e)

    async def ashutdown_sandbox(self, agent: Union[Agent, Team]) -> str:
        """Delete the current sandbox and everything in it.

        Returns:
            A success message or an error message.
        """
        try:
            sandbox_id = self._current_sandbox_id(agent)
            if not sandbox_id:
                return "No active sandbox to shut down."
            await self._arequest("DELETE", f"{API_URL}/instances/{sandbox_id}")
            self._forget_sandbox(agent)
            return f"Sandbox {sandbox_id} shut down."
        except Exception as e:
            return self._error("Error shutting down sandbox", e)
