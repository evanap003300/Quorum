# Physics Problem Solver Orchestrator

A sophisticated AI-powered system for solving physics and mathematics problems through structured planning and step-by-step execution. The orchestrator handles both **numeric solutions** and **symbolic derivations** with automatic cost tracking and performance metrics.

## Features

### 🔢 Dual-Mode Problem Solving
- **Numeric Mode**: Solves problems with concrete values (e.g., "A car accelerates at 2 m/s² for 5 seconds")
- **Symbolic Mode**: Derives symbolic expressions for general solutions (e.g., "A car of mass M is hit by balls at speed u")
- **Mixed Mode**: Supports problems combining both numeric and symbolic variables

### 🧮 Advanced Computation
- Uses **SymPy** for symbolic mathematics (derivatives, integrals, simplification)
- Access to physical constants via:
  - `scipy.constants` (speed of light, Planck constant, gravity, etc.)
  - `astropy.constants` (astronomical constants)
  - `mendeleev` (periodic table data)
- Automatic unit handling and conversions

### 📊 Complete Tracking
- **Timing Metrics**: Planning time, execution time, total time
- **Cost Tracking**: Per-step and total API costs in USD
- **Detailed Logging**: Step-by-step progress with timings and costs

### 🤖 Flexible LLM Integration
- Models can use tools OR respond directly - no forced constraints
- Graceful fallback from code execution to text extraction
- Automatic retry with helpful feedback on failures
- Support for OpenRouter, Gemini, and GPT-4 models

## Architecture

### Three-Phase Problem Solving

```
Problem Input
    ↓
[1] PLANNER (LLM) → Structured Plan with steps
    ↓
[2] SOLVER (LLM + Execution) → Execute each step atomically
    ↓
[3] RESULT → Final answer with statistics
```

### Key Components

#### `planner/planner.py`
- Analyzes problem and creates step-by-step plan
- Detects symbolic vs numeric variables automatically
- Returns `StateObject` (variables) and `Plan` (steps)
- Tracks planning cost

#### `solver/solver.py`
- Executes individual calculation steps
- Supports three operations:
  - **extract**: Pull values from problem text
  - **calculate**: Perform mathematical operations
  - **convert**: Change units
- Conditionally guides model to use SymPy for symbolic steps
- Tracks per-step costs and execution code

#### `orchestrate.py`
- Coordinates planning and execution phases
- Collects timing and cost metrics
- Displays results with statistics
- Handles both numeric and symbolic outputs transparently

## Installation

### Requirements
```bash
pip install pydantic python-dotenv openai sympy pint numpy scipy astropy mendeleev e2b-code-interpreter
```

### Environment Setup
Create a `.env` file in the orchestrator directory:
```env
OPEN_ROUTER_KEY=your_openrouter_api_key_here
```

## Usage

### Basic Example: Numeric Problem

```python
from orchestrate import solve_problem

result = solve_problem("A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?")

if result["success"]:
    print(f"Answer: {result['final_answer']} {result['final_unit']}")
    print(f"Total cost: ${result['total_cost']:.4f}")
    print(f"Total time: {result['total_time']:.2f}s")
```

### Advanced Example: Symbolic Problem

```python
result = solve_problem(
    "A car of mass M is hit by baseballs thrown at speed u at a mass rate of σ kg/s. "
    "Find velocity as a function of time."
)

if result["success"]:
    # Result contains symbolic expression like: M*u*t/(M + sigma*t)
    print(f"v(t) = {result['final_answer']}")
    print(f"Total cost: ${result['total_cost']:.4f}")
```

### Running Tests

```bash
python orchestrate.py
```

This runs the built-in test suite with sample problems.

## Output Example

```
======================================================================
PROBLEM 1
======================================================================
Problem: A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?

============================================================
PLANNING
============================================================
✓ Created plan with 4 steps
  Domain: kinematics
  Approach: Apply the first kinematic equation of motion
  Target: v
  Planning time: 5.23s | Cost: $0.0045

============================================================
EXECUTION
============================================================

Step 1/4: Extract initial velocity from 'from rest'
  Operation: OperationType.EXTRACT
  Output: v0 (m/s)
  ✓ v0 = 0.0 m/s
    Time: 2.34s | Cost: $0.0012

Step 2/4: Extract acceleration value
  Operation: OperationType.EXTRACT
  Output: a (m/s^2)
  ✓ a = 2.0 m/s^2
    Time: 1.89s | Cost: $0.0008

Step 3/4: Extract time duration
  Operation: OperationType.EXTRACT
  Output: t (s)
  ✓ t = 5.0 s
    Time: 2.12s | Cost: $0.0009

Step 4/4: Calculate final velocity using kinematic equation
  Operation: OperationType.CALCULATE
  Output: v (m/s)
  ✓ v = 10.0 m/s
    Time: 3.45s | Cost: $0.0015

============================================================
RESULT
============================================================

Final answer: v = 10.0 m/s

============================================================
STATISTICS
============================================================
Planning time:    5.23s     | Cost: $0.0045
Execution time:   9.80s    | Cost: $0.0044
Total time:       15.03s
Total cost:       $0.0089
Cost per step:    $0.0022
```

## Supported Operations

### Extract
Pulls values from problem text. Automatically handles:
- **Numeric values**: "2 m/s²" → 2.0
- **Symbolic variables**: "mass M" → "M"
- **Physical constants**: Uses `sci_constants.c` for speed of light instead of hardcoding

### Calculate
Performs mathematical operations using:
- **Numeric**: Standard arithmetic via NumPy
- **Symbolic**: SymPy derivatives, integrals, equations, and simplification
- Automatically detects operation type based on input variables

### Convert
Handles unit conversions with:
- **Numeric**: Pint library for standard conversions
- **Symbolic**: Unit-aware calculations (keeps dimensions consistent)

## Pricing

### Model Costs (OpenRouter)
- **GPT-4.1-mini**: $0.4 per 1M input tokens, $1.6 per 1M output tokens
- **Gemini 3 Pro**: $2 per 1M input tokens, $12 per 1M output tokens

All costs are calculated per-step and aggregated for total problem cost.

## How Symbolic Detection Works

The planner automatically determines if each variable is symbolic or numeric:

| Text Pattern | Type | Example |
|--------------|------|---------|
| "value given" | Numeric | "A 5 kg object" |
| "value = number" | Numeric | "mass M = 5 kg" |
| "variable letter" | Symbolic | "mass M" |
| "no value provided" | Symbolic | "thrown at speed u" |
| "Greek letter/symbol" | Symbolic | "rate σ kg/s" |

Steps are marked symbolic if **any** input or output is symbolic. The solver uses SymPy for symbolic steps and automatically simplifies results.

## Schema

### Variable
```python
class Variable(BaseModel):
    name: str                                    # e.g., "M"
    description: str                             # e.g., "car mass"
    expected_unit: str                           # e.g., "kg"
    value: Optional[Union[float, str]] = None    # 5.0 or "M"
    unit: Optional[str] = None                   # e.g., "kg"
    source_step: Optional[int] = None            # Step that computed it
    is_symbolic: bool = False                    # True if symbolic
```

### Step
```python
class Step(BaseModel):
    step_id: int
    operation: OperationType                     # extract, calculate, convert
    description: str                             # What this step does
    inputs: List[str]                            # Input variable names
    output: str                                  # Output variable name
    formula: Optional[str] = None                # Mathematical formula
    expected_unit: str                           # Expected unit of output
    justification: str                           # Why this step is needed
    is_symbolic: bool = False                    # True if symbolic
```

## Troubleshooting

### Model Refuses to Use Tool
The system now handles this gracefully. If the model doesn't use the tool, it:
1. Extracts result from text response
2. Retries with helpful feedback
3. Only errors if all 3 attempts fail

### Symbolic Simplification Issues
Results are automatically simplified with `sp.simplify()`. If you need different simplification:
- Update the planner prompt with specific simplification rules
- Or adjust the solver's `_build_calculate_prompt` function

### Cost Calculation Differs from OpenRouter
Ensure your `.env` has the correct API key and that pricing constants in `planner.py` and `solver.py` match current OpenRouter rates.

## Future Enhancements

- [ ] Support for multi-variable calculus
- [ ] Numerical integration/ODE solving
- [ ] Matrix and linear algebra operations
- [ ] Interactive step verification
- [ ] Problem explanation generation
- [ ] Batch problem solving

