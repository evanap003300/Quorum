import asyncio
from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import Sandbox

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

async def run(code: str) -> str:
    sandbox = Sandbox()

    # Install required libraries
    sandbox.commands.run("pip install sympy pint numpy scipy astropy mendeleev")

    # Load constants libraries in the sandbox
    sandbox.run_code(SETUP_CODE)

    if code:
        execution = sandbox.run_code(code)
        # Combine stdout and stderr to catch all output
        output = "".join(execution.logs.stdout)
        if execution.logs.stderr:
            output += "".join(execution.logs.stderr)
        return output

    return ""

if __name__ == '__main__':
    output = asyncio.run(run("print(sci_constants.c)"))
    print(output)
