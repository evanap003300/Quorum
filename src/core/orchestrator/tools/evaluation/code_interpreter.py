import asyncio
import subprocess
import sys
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# Try to import e2b, but don't fail if it's not available or broken
Sandbox = None
try:
    from e2b_code_interpreter import Sandbox
except Exception:
    pass

# Setup code that runs in every sandbox to load constants libraries
# Made resilient to missing packages
SETUP_CODE = """
import sys
import numpy as np
import sympy as sp
import scipy.constants
from scipy import constants as sci_constants

# Import astropy if available, provide fallback if not
try:
    from astropy import constants as astro_constants
    from astropy import units as u
except ImportError:
    # Provide minimal fallback if astropy is not installed
    class MockAstropy:
        pass
    astro_constants = MockAstropy()
    u = MockAstropy()

# Import mendeleev if available (chemistry support)
try:
    from mendeleev import element
except (ImportError, ModuleNotFoundError):
    pass

# Import pint for unit conversions
try:
    import pint
    ureg = pint.UnitRegistry()
except (ImportError, ModuleNotFoundError):
    ureg = None
"""

async def init_sandbox(sandbox) -> None:
    """Initialize a sandbox with required libraries and imports."""
    if sandbox is None:
        return
    try:
        # Install required libraries
        sandbox.commands.run("pip install sympy pint numpy scipy astropy mendeleev")
        # Load constants libraries in the sandbox
        sandbox.run_code(SETUP_CODE)
    except Exception as e:
        print(f"Warning: Failed to initialize sandbox: {e}")

async def run_in_sandbox(code: str, sandbox) -> str:
    """Execute code in an existing sandbox and return output."""
    if code and sandbox:
        try:
            execution = sandbox.run_code(code)
            # Combine stdout and stderr to catch all output
            output = "".join(execution.logs.stdout)
            if execution.logs.stderr:
                output += "".join(execution.logs.stderr)
            return output
        except Exception:
            pass
    return ""

async def _ensure_packages_installed() -> None:
    """Try to install required packages if they're missing."""
    required_packages = ['numpy', 'sympy', 'scipy', 'astropy', 'pint', 'mendeleev']

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            try:
                # Try to install the package
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-q', package],
                    timeout=30,
                    capture_output=True,
                    check=False
                )
            except Exception:
                # If installation fails, continue with other packages
                # The SETUP_CODE has fallbacks for missing packages
                pass

async def run_local(code: str) -> str:
    """
    Fallback: Execute code locally in the current Python interpreter.
    WARNING: This is less isolated than E2B but works without authentication.
    """
    if not code:
        return ""

    try:
        # Ensure required packages are installed (first time only, minimal overhead)
        await _ensure_packages_installed()

        # Create empty namespace for code execution
        namespace = {}

        # Capture output
        import io
        from contextlib import redirect_stdout, redirect_stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # First, execute SETUP_CODE to initialize all required imports and libraries
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(SETUP_CODE, namespace)

        # Then execute the actual code with all imports available
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)

        output = stdout_capture.getvalue()
        if stderr_capture.getvalue():
            output += stderr_capture.getvalue()

        return output
    except Exception as e:
        return f"Error: {str(e)}"

async def run(code: str, sandbox: Optional[object] = None) -> str:
    """
    Execute code in a sandbox. Creates new sandbox if not provided.

    Args:
        code: Python code to execute
        sandbox: Optional existing sandbox. If None, uses local execution as fallback.

    Returns:
        Output from code execution
    """
    own_sandbox = sandbox is None

    if sandbox:
        try:
            output = await run_in_sandbox(code, sandbox)
            if output or not own_sandbox:
                return output
        except Exception:
            pass

    # Fallback to local execution
    return await run_local(code)

if __name__ == '__main__':
    output = asyncio.run(run("print(sci_constants.c)"))
    print(output)
