import asyncio
from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import Sandbox

async def run(code: str) -> str:
    sandbox = Sandbox()

    sandbox.commands.run("pip install sympy pint numpy")

    if code:
        execution = sandbox.run_code(code)
        # Combine stdout and stderr to catch all output
        output = "".join(execution.logs.stdout)
        if execution.logs.stderr:
            output += "".join(execution.logs.stderr)
        return output

    return ""

if __name__ == '__main__':
    output = asyncio.run(run("print('Hello world!')"))
    print(output)
