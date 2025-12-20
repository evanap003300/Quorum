# Quorum — AI Physics & Mathematics Problem Solver

An intelligent, structured problem-solving system that decomposes complex physics and mathematics problems into atomic steps, generates verified code for each step, and maintains accuracy through rigorous state tracking and error detection.

## Overview

Quorum uses a multi-step approach to solve physics and mathematics problems with high accuracy:

1. **Plan** - Breaks down complex problems into atomic, executable steps
2. **Execute** - Generates and runs Python code for each step in a sandboxed environment
3. **Track** - Maintains state throughout the solving process to minimize hallucinations and errors
4. **Verify** - Validates results and detects degradation or accuracy loss

The system uses structured JSON state objects to track assumptions, known values, and intermediate results, combined with MARS review planning methodology to ensure each step is logically sound and correctly executed.

## How It Works

### Step 1: Planning
When you provide a problem, the **Planner** breaks it down into atomic steps:

- Uses Google Gemini 3 Pro to understand the problem structure
- Identifies all known values and unknowns
- Creates a step-by-step plan where each step performs exactly ONE operation
- Each step is one of three types:
  - **Extract**: Read a value directly from the problem
  - **Calculate**: Perform a mathematical operation
  - **Convert**: Convert between units

Example breakdown for "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?":
```
Step 1: Extract v0 = 0 (at rest)
Step 2: Extract a = 2 m/s²
Step 3: Extract t = 5 seconds
Step 4: Calculate v = v0 + a*t = 0 + 2*5 = 10 m/s
```

### Step 2: Execution
For each step, the **Solver** generates and executes Python code:

- Generates executable Python code based on the step requirements
- Runs code in a sandboxed E2B environment for security
- Has access to: `numpy`, `sympy`, `pint` (for unit handling)
- Supports up to 3 retry attempts if code generation fails
- Extracts the result in the format: `<value> <unit>`

### Step 3: State Tracking
The **Orchestrator** maintains a complete state object throughout execution:

- Tracks all variables: given values, calculated intermediates, final answers
- Records which step calculated each value
- Maintains unit information for each variable
- Enables verification of calculation chains

### Step 4: Final Answer
Once all steps complete, the system returns:

- The final calculated value with units
- Complete state history
- The full plan that was executed
- Any errors encountered during execution

## Architecture

```
┌─────────────────┐
│  User Problem   │
└────────┬────────┘
         │
    ┌────▼────────────────────┐
    │  orchestrate.solve()    │ (Main entry point)
    │  - Coordinates workflow │
    └────┬────────────────────┘
         │
         ├──────────────────┐
         │                  │
    ┌────▼─────────┐  ┌────▼──────────┐
    │  Planner     │  │  Solver       │
    │ - Creates    │  │  - Executes   │
    │   atomic     │  │    each step  │
    │   steps      │  │  - Generates  │
    │              │  │    code       │
    └──────────────┘  └────┬──────────┘
                           │
                      ┌────▼────────────┐
                      │  E2B Sandbox    │
                      │ - Runs Python   │
                      │   safely        │
                      └─────────────────┘
         │
    ┌────▼──────────────────┐
    │  Final Answer + State │
    │  - Value & unit       │
    │  - Full history       │
    │  - Plan executed      │
    └───────────────────────┘
```

## Core Components

### 1. Orchestrator (`src/core/orchestrator/orchestrate.py`)
**Coordinates the entire problem-solving pipeline**

Main function: `solve_problem(problem: str)`
- Calls planner to create a plan
- Iterates through each step
- Calls solver for each step
- Updates state with results
- Returns final answer or error

### 2. Planner (`src/core/orchestrator/planner/planner.py`)
**Decomposes problems into atomic steps**

Main function: `plan(problem: str) -> Tuple[StateObject, Plan]`
- Uses Google Gemini 3 Pro (via OpenRouter)
- Extracts key values from the problem
- Creates step-by-step solution approach
- Returns structured plan and initial state

### 3. Solver (`src/core/orchestrator/solver/solver.py`)
**Executes individual atomic steps**

Main function: `solve_step(step: Step, state: StateObject) -> Tuple[bool, Optional[float], Optional[str], Optional[str]]`
- Generates Python code for each step
- Executes code in E2B sandbox
- Parses output to extract value and unit
- Returns success status and result

### 4. Data Models (`src/core/orchestrator/planner/schema.py`)
**Defines the structure for plans and state**

Key classes:
- `Variable` - Represents a single variable with name, unit, value, and source
- `StateObject` - Encapsulates problem state and assumptions
- `Step` - Atomic operation with inputs, formula, and expected output
- `Plan` - Complete solution strategy with all steps

### 5. Python Interpreter (`src/core/orchestrator/solver/python_interpreter-e2b/main.py`)
**Provides sandboxed code execution**

- Uses E2B Code Interpreter for safe, isolated Python execution
- Automatically installs required libraries
- Captures and returns code output

### 6. Format Validation (`src/core/orchestrator/degradation/format.py`)
**Detects degradation and hallucinations** (future enhancement)

## Setup & Configuration

### Requirements

```bash
# Core dependencies
pydantic>=2.5.0           # Data validation
python-dotenv>=1.0.0      # Environment configuration
openai>=1.0.0             # LLM API client

# Code execution
e2b-code-interpreter>=1.5.2  # Sandboxed execution
numpy>=1.24.0             # Numerical computing
sympy>=1.12               # Symbolic math
pint>=0.23.0              # Unit conversion

# Development
pytest>=7.4.0             # Testing
mypy>=1.7.0               # Type checking
black>=23.12.0            # Code formatting
```

### Environment Variables

Create a `.env` file in the project root:

```
OPEN_ROUTER_KEY=<your_openrouter_api_key>
E2B_API_KEY=<your_e2b_api_key>
```

**Required Keys:**
- **OPEN_ROUTER_KEY**: API key for OpenRouter (provides access to Google Gemini 3 Pro)
- **E2B_API_KEY**: API key for E2B sandboxed code execution

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Test the system
python -m src.core.orchestrator.orchestrate
```

## Usage

### Basic Example

```python
from src.core.orchestrator.orchestrate import solve_problem

problem = "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?"
result = solve_problem(problem)

if result['success']:
    print(f"Answer: {result['final_answer']} {result['final_unit']}")
    print(f"Plan: {result['plan']}")
    print(f"State: {result['state']}")
else:
    print(f"Error: {result['error']}")
```

### Return Value Structure

```python
{
    'success': bool,                  # Whether solving succeeded
    'final_answer': float,            # The calculated answer
    'final_unit': str,                # Unit of the answer
    'state': StateObject,             # Complete state history
    'plan': Plan,                     # The plan that was executed
    'error': Optional[str]            # Error message if unsuccessful
}
```

## Key Features

- **Atomic Steps**: Each step performs exactly one operation for clarity and debuggability
- **Structured Planning**: Uses LLM to intelligently decompose problems
- **Sandboxed Execution**: All code runs in E2B environment for security
- **State Tracking**: Maintains complete history of values and assumptions
- **Error Handling**: Up to 3 retry attempts per step if code generation fails
- **Unit Handling**: Built-in support for unit conversion using Pint
- **Reproducibility**: Full plan and state are returned with results

## Roadmap

- Implement MDAP (multi-step degradation analysis process) for error reduction
- Complete format checker for hallucination detection
- Benchmarking suite for physics and mathematics problems
- FastAPI REST endpoint for HTTP access
- Support for more complex domains (kinematics, energy, forces, etc.)

## Project Structure

```
accurate_problem_solver/
├── src/
│   └── core/
│       ├── main.py                    # Entry point (in development)
│       └── orchestrator/
│           ├── orchestrate.py         # Main orchestrator
│           ├── planner/
│           │   ├── planner.py         # Problem planner
│           │   └── schema.py          # Data models
│           ├── solver/
│           │   ├── solver.py          # Step solver
│           │   └── python_interpreter-e2b/
│           │       └── main.py        # E2B integration
│           └── degradation/
│               └── format.py          # Validation (future)
├── requirements.txt
├── README.md
└── .env
```

## Notes

- The system is optimized for physics and mathematics problems
- Planner uses temperature=0.1 for consistency and reduced hallucinations
- All mathematical operations use SI units or standard physics units
- Code generation and execution happen dynamically based on problem requirements
