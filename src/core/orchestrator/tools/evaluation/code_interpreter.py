import asyncio
from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import Sandbox
from typing import Optional

# Setup code that runs in every sandbox to load constants libraries
SETUP_CODE = """
import numpy as np
import sympy as sp
import scipy.constants
from scipy import constants as sci_constants
from astropy import constants as astro_constants
from astropy import units as u
from mendeleev import element
"""

async def init_sandbox(sandbox: Sandbox) -> None:
    """Initialize a sandbox with required libraries and imports."""
    # Install required libraries
    sandbox.commands.run("pip install sympy pint numpy scipy astropy mendeleev")
    # Load constants libraries in the sandbox
    sandbox.run_code(SETUP_CODE)

async def run_in_sandbox(code: str, sandbox: Sandbox) -> str:
    """Execute code in an existing sandbox and return output."""
    if code:
        execution = sandbox.run_code(code)
        # Combine stdout and stderr to catch all output
        output = "".join(execution.logs.stdout)
        if execution.logs.stderr:
            output += "".join(execution.logs.stderr)
        return output
    return ""

async def run(code: str, sandbox: Optional[Sandbox] = None) -> str:
    """
    Execute code in a sandbox. Creates new sandbox if not provided.

    Args:
        code: Python code to execute
        sandbox: Optional existing sandbox. If None, creates a new one.

    Returns:
        Output from code execution
    """
    own_sandbox = sandbox is None

    if own_sandbox:
        sandbox = Sandbox()
        await init_sandbox(sandbox)

    output = await run_in_sandbox(code, sandbox)

    if own_sandbox:
        sandbox.kill()

    return output

if __name__ == '__main__':
    output = asyncio.run(run("print(sci_constants.c)"))
    print(output)
