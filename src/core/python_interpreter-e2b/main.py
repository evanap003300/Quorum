from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import Sandbox

async def run(code: str) -> None:
    sandbox = Sandbox() # By default the sandbox is alive for 5 minutes
    
    await sandbox.commands.run("sympy")
    await sandbox.install_package("pint")
    await sandbox.install_package("numpy")

    execution = sandbox.run_code("print('hello world')") # Execute Python inside the sandbox
    print(execution.logs)
    execution = sandbox.run_python(code)

    print("Output:", execution.output)
