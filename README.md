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

### Step 0: Image Analysis (Optional)
If an image is provided, the **Vision Module** extracts the problem using LLM-powered CV tool calling:

- Uses GPT-4o with 8 specialized computer vision tools
- LLM iteratively preprocesses images: applies grid, crops regions, enhances clarity
- Solves spatial hallucination with 10×10 grid system (A0-J9 labels)
- Saves intermediate images for debugging
- Returns extracted problem text + diagram context

**Available CV Tools:**
- `get_image_metadata` - Check resolution and viability
- `detect_content_regions` - Find text blocks and diagrams
- `detect_shadows_and_artifacts` - Diagnose quality issues
- `apply_grid` - Overlay navigation grid with labels
- `crop_grid_square` - Zoom to grid cell (e.g., "D5")
- `crop_quadrant` - Quick 25% cropping
- `binarize_image` - Remove shadows, B&W conversion
- `enhance_clarity` - Boost contrast and sharpen

### Step 1: Physics Review
Once a plan is created, the **Physics Lawyer** audits it for conceptual errors:

- Checks for reference frame violations (e.g., absolute vs relative velocity)
- Validates variable mass systems (F = dp/dt for rockets)
- Ensures conservation laws are applied correctly
- Detects small-angle approximation violations
- Validates unit consistency

If errors are found, the **Revisor** automatically repairs the plan while preserving dependencies. This prevents wasted computation on fundamentally flawed approaches.

### Step 2: Planning
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

### Step 3: Execution
For each step, the **Solver** generates and executes Python code:

- Generates executable Python code based on the step requirements
- Runs code in a sandboxed E2B environment for security
- Has access to: `numpy`, `sympy`, `pint` (for unit handling)
- Supports up to 3 retry attempts if code generation fails
- Extracts the result in the format: `<value> <unit>`

### Step 4: State Tracking
The **Orchestrator** maintains a complete state object throughout execution:

- Tracks all variables: given values, calculated intermediates, final answers
- Records which step calculated each value
- Maintains unit information for each variable
- Enables verification of calculation chains

### Step 5: Final Answer
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

### 0. Vision Module (`src/core/orchestrator/vision.py`)
**Extracts problems from images using LLM-powered CV tool calling**

Main function: `analyze_problem_image_with_cv_tools(image_path: str)`
- Calls GPT-4o with 8 computer vision tools
- LLM iteratively preprocesses images before extraction
- Saves intermediate images for debugging
- Returns: (problem_text, diagram_context, cost, intermediate_paths)

**Key features:**
- Tool calling loop: LLM decides which CV tools to apply
- Resolution cliff solution: 10×10 grid with alphanumeric labels (A0-J9)
- Spatial hallucination fix: LLM chains `apply_grid()` → `crop_grid_square()`
- Cost tracking: Accumulates vision image costs per iteration
- Intermediate image saving: Preserves all preprocessing steps

**Tools available:**
1. `get_image_metadata` - Resolution assessment
2. `detect_content_regions` - Find text/diagram blocks
3. `detect_shadows_and_artifacts` - Diagnose issues
4. `apply_grid` - Add navigation reference
5. `crop_grid_square` - Zoom to cell
6. `crop_quadrant` - Quick cropping
7. `binarize_image` - B&W conversion
8. `enhance_clarity` - Contrast + sharpen

### 1. Orchestrator (`src/core/orchestrator/orchestrate.py`)
**Coordinates the entire problem-solving pipeline**

Main function: `solve_problem(problem: str, image_path: Optional[str])`
- Calls vision module if image provided (Step 0)
- Calls planner to create a plan (Step 2)
- Calls physics lawyer + revisor for validation (Step 1)
- Iterates through each step
- Calls solver for each step (Step 3)
- Updates state with results
- Returns final answer or error

### 2. Planner (`src/core/orchestrator/planner/planner.py`)
**Decomposes problems into atomic steps**

Main function: `plan(problem: str) -> Tuple[StateObject, Plan]`
- Uses Google Gemini 3 Pro (via Google Generative AI)
- Extracts key values from the problem
- Creates step-by-step solution approach
- Returns structured plan and initial state
- Prompt template: `prompts/planning.py:PLANNER_PROMPT`

### 3. Physics Lawyer (`src/core/orchestrator/planner/critics/physics_lawyer.py`)
**Audits plans for conceptual physics errors**

Main function: `audit_plan(problem: str, plan: Plan) -> AuditResult`
- Uses Google Gemini 3 Pro for auditing
- Checks 7 categories of physics violations:
  1. Reference Frame errors
  2. Variable Mass systems
  3. Conservation Law violations
  4. Approximation errors
  5. Unit inconsistencies
  6. Physical constraint violations
  7. Dependency issues
- Returns JSON-structured audit results
- Prompt template: `prompts/critics.py:PHYSICS_LAWYER_PROMPT`

### 4. Revisor (`src/core/orchestrator/planner/revisor.py`)
**Repairs flagged plans while preserving dependencies**

Main function: `revise_plan(problem: str, plan: Plan, critiques: List) -> Plan`
- Uses Google Gemini 3 Pro for plan revision
- Surgically fixes flagged steps
- Preserves variable names and units for downstream steps
- Can insert new intermediate steps if needed
- Returns corrected plan in same JSON schema
- Prompt template: `prompts/revisor.py:REVISOR_PROMPT`

### 5. Solver (`src/core/orchestrator/solver/solver.py`)
**Executes individual atomic steps** (slim entry point, ~65 lines)

Main function: `solve_step(step: Step, state: StateObject, sandbox: Optional[Sandbox])`
- Routes step execution based on operation type
- Delegates to execution and parsing modules
- Passes hot sandbox for reuse across steps
- Returns success status and result

#### 5a. Solver Execution (`src/core/orchestrator/solver/execution.py`)
**LLM-powered code generation and execution** (~250 lines)

Functions:
- `execute_with_llm()` - Single-output step execution
- `execute_with_llm_multi_output()` - Batch extraction execution
- Uses gpt-4.1-mini-2025-04-14 for code generation
- Supports up to 3 retry attempts if generation fails

#### 5b. Solver Parsing (`src/core/orchestrator/solver/parsing.py`)
**Output processing and validation** (~200 lines)

Functions:
- `parse_output()` - Extracts value and unit from execution output
- `parse_multi_output()` - Handles batch extraction results
- `extract_result_from_text()` - Fallback text extraction
- `validate_result()` - Validates results meet semantic requirements
- `calculate_cost()` - Computes API token costs

#### 5c. Solver Prompts (`src/core/orchestrator/prompts/solver.py`)
**Prompt templates and builders** (~120 lines)

Constants:
- `SOLVER_SYSTEM_MESSAGE` - System message for single-output steps
- `SOLVER_MULTI_OUTPUT_SYSTEM_MESSAGE` - System message for batch steps

Functions:
- `build_extract_prompt()` - Generates extraction prompts
- `build_calculate_prompt()` - Generates calculation prompts
- `build_convert_prompt()` - Generates unit conversion prompts

### 6. Data Models (`src/core/orchestrator/planner/schema.py`)
**Defines the structure for plans and state**

Key classes:
- `Variable` - Represents a single variable with name, unit, value, and source
- `StateObject` - Encapsulates problem state and assumptions
- `Step` - Atomic operation with inputs, formula, and expected output
- `Plan` - Complete solution strategy with all steps

### 7. Tools (`src/core/orchestrator/tools/`)
**Specialized tool suites for image preprocessing and code execution**

#### 7a. Computer Vision Tools (`src/core/orchestrator/tools/vision/`)
**18 specialized CV tools organized into 4 suites:**

**Spatial Suite** (navigation & focus):
- `crop_quadrant()` - Quick 25% cropping
- `crop_region()` - Precise pixel cropping
- `apply_grid()` - 10×10 navigation grid
- `crop_grid_square()` - Zoom to grid cell

**Clarity Suite** (enhancement & restoration):
- `binarize_image()` - Adaptive B&W conversion (removes shadows)
- `invert_colors()` - Fix blackboard/dark-mode images
- `enhance_clarity()` - Boost contrast + sharpen
- `stretch_contrast()` - Full dynamic range
- `denoise_image()` - Noise reduction
- `apply_unsharp_mask()` - Targeted sharpening

**Debugging Suite** (metadata & analysis):
- `get_image_metadata()` - Resolution assessment
- `analyze_image_contrast()` - Brightness/contrast metrics
- `detect_shadows_and_artifacts()` - Problem diagnosis
- `compare_images()` - A/B testing preprocessing

**Content Detection Suite** (automatic region finding):
- `detect_content_regions()` - Find text/diagram blocks
- `detect_text_regions()` - Locate text areas
- `detect_diagram_regions()` - Find visual content
- `highlight_content_regions()` - Visualization overlay

#### 7b. Code Interpreter (`src/core/orchestrator/tools/evaluation/code_interpreter.py`)
**Sandboxed Python execution using E2B**

Features:
- Hot sandbox reuse for 60-90% performance improvement
- Automatic library installation (numpy, sympy, pint)
- Safe, isolated execution environment
- Cost-effective per-step execution

### 8. Configuration (`src/core/orchestrator/config/pricing.py`)
**Centralized model pricing configuration**

- Single source of truth for model costs
- Consolidates pricing from all modules
- Supports both direct OpenAI and OpenRouter models

### 9. Prompts (`src/core/orchestrator/prompts/`)
**Centralized prompt templates** (~700 lines total)

Modules:
- `prompts/vision.py` - Vision analysis prompts (VISION_PROMPT, ENHANCED_VISION_PROMPT)
- `prompts/planning.py` - Problem planning prompt (PLANNER_PROMPT)
- `prompts/solver.py` - Solver prompts and builders
- `prompts/critics.py` - Physics Lawyer audit prompt (PHYSICS_LAWYER_PROMPT)
- `prompts/revisor.py` - Plan revisor prompt (REVISOR_PROMPT)

### 10. Format Validation (`src/core/orchestrator/degradation/format.py`)
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
OPENAI_API_KEY=<your_openai_api_key>           # Required, for GPT-4o vision and code generation
GOOGLE_API_KEY=<your_google_api_key>           # Required, for Google Gemini 3 Pro (planning/review)
E2B_API_KEY=<your_e2b_api_key>                 # Required, for E2B sandboxed code execution
```

**Required Keys:**
- **OPENAI_API_KEY**: API key for OpenAI (GPT-4o for vision + code generation, gpt-4.1-mini for solver)
- **GOOGLE_API_KEY**: API key for Google Generative AI (Gemini 3 Pro for planning, Gemini 3 Flash for revisor)
- **E2B_API_KEY**: API key for E2B sandboxed code execution

**Environment Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
echo "OPENAI_API_KEY=sk-..." >> .env
echo "GOOGLE_API_KEY=AIza..." >> .env
echo "E2B_API_KEY=..." >> .env
```

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
- **Physics Validation**: Physics Lawyer audits plans for conceptual errors before execution
- **Automatic Repair**: Revisor automatically fixes flagged plans while preserving dependencies
- **Sandboxed Execution**: All code runs in E2B environment for security
- **State Tracking**: Maintains complete history of values and assumptions
- **Error Handling**: Up to 3 retry attempts per step if code generation fails
- **Unit Handling**: Built-in support for unit conversion using Pint
- **Reproducibility**: Full plan and state are returned with results

## Roadmap

- ✅ **Physics Lawyer & Revisor**: Conceptual error detection and automatic repair (COMPLETE)
- Implement MDAP (multi-step degradation analysis process) for execution-time error reduction
- Complete format checker for hallucination detection
- Domain-specific critics (thermodynamics, quantum mechanics, etc.)
- Interactive correction approval (allow user to review and approve fixes)
- Benchmarking suite for physics and mathematics problems
- FastAPI REST endpoint for HTTP access
- Support for more complex domains (kinematics, energy, forces, etc.)

## Project Structure

```
accurate_problem_solver/
├── src/
│   └── core/
│       ├── main.py                      # Entry point (in development)
│       └── orchestrator/
│           ├── orchestrate.py           # Main orchestrator
│           │
│           ├── config/                  # Configuration
│           │   ├── __init__.py
│           │   └── pricing.py           # Unified model pricing
│           │
│           ├── prompts/                 # Centralized prompts
│           │   ├── __init__.py
│           │   ├── vision.py            # Vision analysis prompts
│           │   ├── planning.py          # Planning prompt
│           │   ├── solver.py            # Solver prompts & builders
│           │   ├── critics.py           # Physics Lawyer prompt
│           │   └── revisor.py           # Revisor prompt
│           │
│           ├── planner/
│           │   ├── planner.py           # Problem planner
│           │   ├── schema.py            # Data models
│           │   ├── critics/
│           │   │   ├── __init__.py
│           │   │   └── physics_lawyer.py # Physics auditor
│           │   └── revisor.py           # Plan repair
│           │
│           ├── solver/
│           │   ├── solver.py            # Main entry point
│           │   ├── execution.py         # LLM executors
│           │   └── parsing.py           # Output parsing
│           │
│           ├── tools/
│           │   ├── vision/              # Computer Vision tools
│           │   │   ├── __init__.py
│           │   │   ├── utils.py         # Helper functions
│           │   │   ├── spatial.py       # Cropping & grid tools
│           │   │   ├── clarity.py       # Enhancement tools
│           │   │   ├── debugging.py     # Analysis & diagnostics
│           │   │   ├── content_detection.py # Auto region finding
│           │   │   └── README.md        # CV tools documentation
│           │   └── evaluation/
│           │       ├── __init__.py
│           │       └── code_interpreter.py # E2B sandbox wrapper
│           │
│           ├── testing/
│           │   ├── __init__.py
│           │   ├── test_vision_debug.py
│           │   ├── test_vision_integration.py
│           │   └── test_cv_integration.py
│           │
│           ├── vision.py                # Vision API with tool calling
│           └── degradation/
│               └── format.py            # Validation (future)
│
├── requirements.txt
├── README.md
├── CRITIC_REVISOR_ARCHITECTURE.md      # Detailed architecture
└── .env
```

### Code Organization Improvements

The codebase has been refactored for better maintainability:

- **Prompts centralized**: All prompt templates in `prompts/` directory for easy editing
- **Config consolidation**: Pricing and configuration in `config/` for DRY principle
- **Modular solver**: Solver split into logical concerns:
  - `solver.py` - Clean entry point (~65 lines)
  - `execution.py` - LLM tool loops and code generation (~250 lines)
  - `parsing.py` - Output parsing and validation (~200 lines)
- **No performance impact**: Python imports are cached, zero ongoing overhead
- **Clean architecture**: Clear separation of concerns, easier to test and maintain

## Performance & Optimization

### Model Configuration

- **Planning**: Gemini 3 Pro (via Google Generative AI) - optimized for problem decomposition
- **Physics Review**: Gemini 3 Flash (via Google Generative AI) - optimized for auditing and revision
- **Solving**: GPT-4.1-mini-2025-04-14 (via OpenAI) - optimized for code generation and variable naming
- **Vision**: GPT-4o (via OpenAI) - advanced vision capabilities + tool calling for image analysis
- **Temperature**: 0.1 for planning/vision, 0.0 for revision - low randomness ensures consistency

### Performance Features

- **Hot Sandbox**: Reuses E2B sandbox across all steps (~70% faster per step)
- **Direct OpenAI API**: Uses direct OpenAI connection when available (2-3s faster per step)
- **Batch Extraction**: Groups multiple variable extractions into single steps
- **Cost Tracking**: Real-time USD cost calculation for all API calls

### Expected Performance

- **Per-step solving time**: 4-7 seconds (with hot sandbox and direct API)
- **Per-image vision analysis**: 20-90 seconds with CV tools (simple extraction ≈20s, complex preprocessing ≈90s)
- **CV tool preprocessing**: 0-10 iterations max (typically 2-5 tools used)
- **Physics review time**: 3-8 seconds per audit pass
- **Overall pipeline**: ~1-2 minutes for simple problem, 3-5 minutes for complex multi-step physics problems

## Architecture & Design Decisions

### Why Modular Prompts?

Moving prompts to dedicated files (`prompts/` directory) provides:
- Easy prompt editing without touching logic code
- Version control benefits (cleaner diffs for prompt changes)
- Reusability across modules
- Clear separation of concerns

### Why Split Solver.py?

The original 740-line `solver.py` has been split into:
- **`solver.py`** (65 lines): Clean entry point and routing
- **`execution.py`** (250 lines): LLM tool loops and code execution
- **`parsing.py`** (200 lines): Output processing and validation

This improves:
- Code readability and maintainability
- Testability (can unit test each component)
- Reusability (parsing functions can be used elsewhere)

### Why Centralized Pricing?

`config/pricing.py` consolidates model pricing:
- Single source of truth (no duplication)
- Easy to update when model costs change
- Supports multiple models and APIs

## Notes

- The system is optimized for physics and mathematics problems
- All mathematical operations use SI units or standard physics units
- Code generation and execution happen dynamically based on problem requirements
- Refactored for code clarity with zero performance impact (Python imports are cached)
