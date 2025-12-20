import asyncio
from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import Sandbox

async def run(code: str) -> str:
    sandbox = Sandbox()

    sandbox.commands.run("pip install sympy pint numpy")

    if code:
        execution = sandbox.run_code(code)
        return "".join(execution.logs.stdout)

    return ""

if __name__ == '__main__':
    output = asyncio.run(run("print('Hello world!')"))
    print(output)
