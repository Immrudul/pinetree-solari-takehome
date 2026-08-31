import os

from dotenv import load_dotenv
from solari_sandbox import SandboxClient

from src.trace import TraceLogger


load_dotenv()


class SolariSandbox:
    def __init__(self, trace_logger: TraceLogger | None = None):
        self.client = SandboxClient(
            api_key=os.environ["SOLARI_API_KEY"],
            base_url="https://api.getsolari.com",
        )

        self.sandbox = None
        self.repo_dir = None
        self.trace_logger = trace_logger

    async def start(self):
        print("Creating Solari sandbox...")

        self.sandbox = await self.client.create(
            template="base",
        )

        await self.sandbox.connect()

        print("Sandbox created and connected.")

    async def run(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
    ):
        if self.sandbox is None:
            raise RuntimeError("Sandbox has not been started.")

        args = args or []

        try:
            result = await self.sandbox.commands.run(
                command,
                args=args,
                cwd=cwd,
            )
        except Exception as error:
            message = str(error)
            reconnectable = (
                "Not connected" in message
                or "Control channel closed" in message
                or "1006" in message
            )

            if not reconnectable:
                raise

            print("Solari connection dropped. Reconnecting...")
            await self.sandbox.connect()

            result = await self.sandbox.commands.run(
                command,
                args=args,
                cwd=cwd,
            )

        if self.trace_logger:
            self.trace_logger.record_command(
                command=command,
                args=args,
                cwd=cwd,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exitCode,
            )

        return result

    async def clone_repo(self, repo_url: str):
        self.repo_dir = "/workspace/repo"

        print(f"Cloning {repo_url}...")

        result = await self.run(
            "git",
            ["clone", repo_url, self.repo_dir],
        )

        if result.exitCode != 0:
            raise RuntimeError(
                f"Failed to clone repo:\n{result.stderr}"
            )

        print("Repo cloned successfully.")

        return self.repo_dir

    async def run_in_repo(
        self,
        command: str,
        args: list[str] | None = None,
    ):
        if self.repo_dir is None:
            raise RuntimeError("No repo has been cloned yet.")

        return await self.run(
            command,
            args=args,
            cwd=self.repo_dir,
        )

    async def stop(self):
        if self.sandbox is not None:
            await self.sandbox.kill()
            self.sandbox = None
