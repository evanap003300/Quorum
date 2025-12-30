# Quorum — AI Physics & Mathematics Problem Solver

An intelligent, structured problem-solving system that decomposes complex physics and mathematics problems into atomic steps, generates verified code for each step, and maintains accuracy through rigorous state tracking and error detection.

## Overview

Quorum uses a multi-step approach to solve physics and mathematics problems with high accuracy:

1. **Plan** - Breaks down complex problems into atomic, executable steps
2. **Execute** - Generates and runs Python code for each step in a sandboxed environment
3. **Track** - Maintains state throughout the solving process to minimize hallucinations and errors
4. **Verify** - Validates results and detects degradation or accuracy loss

The system uses structured JSON state objects to track assumptions, known values, and intermediate results, combined with MARS review planning methodology to ensure each step is logically sound and correctly executed.

## 🚦 Smart Routing System (NEW)

Quorum now includes an intelligent **3-tier routing system** that classifies problems by difficulty and dispatches them to the most cost-effective solver:

### Tier Architecture

| Tier | Problem Type | Solver | Model | Speed | Cost | Use Case |
|------|-------------|--------|-------|-------|------|----------|
| **EASY** | Simple definitions, single-step math, unit conversions | Single-Agent | `gemini-3-flash-preview` | 2-3x faster | 8-10x cheaper | Trivial problems, quick answers |
| **MEDIUM** | Textbook problems (AP Physics, Calc I-II), 2-4 clear steps | Single-Agent | `gemini-3-pro-preview` | Standard | Standard | Standard homework/textbook problems |
| **HARD** | SciBench, ambiguous problems, multi-page derivations | Multi-Agent + Swarm | `gemini-3-pro-preview` | Standard | Standard | Complex problems requiring planning & validation |

### How Routing Works

1. **Classification**: When a problem arrives, the router uses `gemini-3-flash-preview` to classify it into EASY/MEDIUM/HARD
2. **Dispatch**: Based on classification, the problem is routed to the appropriate solver:
   - **EASY** → Single-agent solver with Flash (fast & cheap, no planning overhead)
   - **MEDIUM** → Single-agent solver with Pro (one-shot solving, no step decomposition)
   - **HARD** → Full multi-agent orchestrator (planning + swarm + validation)
3. **Execution**: The appropriate solver executes with proper model and parameters
4. **Result Tracking**: Routing metadata (tier, confidence, cost) is captured in results for analysis

### Expected Benefits

- **Cost Reduction**: 30-50% overall cost savings (if 40%+ problems are EASY)
- **Speed Improvement**: 15-25% faster execution
- **Smart Resource Allocation**: Don't waste orchestrator cycles on simple problems
- **Router Overhead**: ~$0.0008 per problem + 0.5-1 second (negligible)

### Classification Criteria

**EASY Tier:**
- Simple concept recall ("What is Newton's Second Law?")
- One-step arithmetic ("If F=ma, m=5kg, a=2m/s², find F")
- Direct unit conversions ("Convert 72 km/h to m/s")
- Looking up constants

**MEDIUM Tier:**
- Projectile motion, circuits, oscillators
- 2-4 clearly defined steps
- Familiar patterns but some algebra/calculus
- Solvable in 3-7 minutes by competent student

**HARD Tier:**
- Advanced physics (quantum, relativity, statistical mechanics)
- Ambiguous problem statements requiring interpretation
- Multi-page derivations or proofs
- Multiple competing approaches needed
- SciBench-level problems
- Problems requiring spatial reasoning from diagrams

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

### Step 3: Execution with K-Ahead Swarm
For each step, the **Solver** executes using K-Ahead Swarm - 3 parallel agents with majority voting:

- **Parallel Execution**: Launches 3 independent agents in parallel for each step
- **Swarm Consensus**: Results voted by numeric proximity (1% tolerance) to eliminate hallucinations
- **Resilient to Errors**: If 1/3 agents crash with syntax error, other 2 still succeed
- **Retry Loop**: Each step retried up to 3 times if swarm fails (up to 9 total agent attempts)
- **Sandboxed Execution**: All code runs in E2B environment for security
- **Libraries**: `numpy`, `sympy`, `pint` (for unit handling)
- **Degree Convention**: Angles output in degrees by default (no radian/degree confusion)
- **Result Format**: `<value> <unit>`

**Swarm Flow:**
```
Step → Swarm Attempt 1 (3 agents parallel)
        ├─ Agent 1 generates code
        ├─ Agent 2 generates code
        └─ Agent 3 generates code
           → Majority vote on results

If all fail → Swarm Attempt 2 (3 agents parallel)
If all fail → Swarm Attempt 3 (3 agents parallel)
If all fail → Step error
```

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
    │  orchestrate.solve()    │ (Main entry point - ASYNC)
    │  - Coordinates workflow │
    └────┬────────────────────┘
         │
         ├──────────────────┐
         │                  │
    ┌────▼─────────┐  ┌────▼──────────────────┐
    │  Planner     │  │  K-Ahead Swarm       │ (NEW)
    │ - Creates    │  │  - 3 agents parallel │
    │   atomic     │  │  - Majority voting   │
    │   steps      │  │  - Retry up to 3x    │
    │              │  └────┬──────────────────┘
    └──────────────┘       │
                      ┌────▼─────────────────┐
                      │  Solver (3 parallel) │
                      │  - Generates code    │
                      │  - ASYNC execution   │
                      └────┬─────────────────┘
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

### 1.5. Router (`src/core/router.py`) 🆕
**Intelligent problem difficulty classification and dispatch** (~185 lines)

Main function: `classify_problem(problem: str, image_path: Optional[str]) -> Tuple[ProblemClassification, float]`
- Uses `gemini-3-flash-preview` for fast, cheap classification
- Classifies problems into EASY/MEDIUM/HARD tiers
- Returns structured output with confidence score and reasoning
- Negligible overhead: ~0.5-1 second, ~$0.0008 per problem

**Key features:**
- **Structured Output**: Pydantic model with tier, confidence, reasoning, key_indicators
- **Cost Tracking**: Calculates routing cost using centralized pricing config
- **Error Handling**: Falls back to MEDIUM tier on router failure (safe middle ground)
- **JSON Schema**: Uses model_name="gemini-3-flash-preview" with response_mime_type="application/json"

**Classification Components:**
- `DifficultyTier` enum: EASY, MEDIUM, HARD
- `ProblemClassification` model: tier, confidence (0-1), reasoning, key_indicators
- `ROUTER_PROMPT` constant: 120-line prompt with detailed tier definitions and examples
- `classify_problem()` function: Entry point for classification

**Integration with Dispatch:**
In `src/benchmarking/runner/problem_executor.py`:
- Called by `_execute_with_timeout()` before problem dispatch
- Metadata added to results for analysis and debugging
- Routing can be disabled with `USE_ROUTING=false` env var for backward compatibility

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

### 5. K-Ahead Swarm (`src/core/orchestrator/solver/swarm.py`) 🆕
**Parallel agent execution with majority voting** (~180 lines)

Main function: `solve_step_with_swarm(step: Step, state: StateObject, sandbox: Optional[Sandbox], k: int = 3)`
- Launches k=3 independent solve_step executions in parallel
- Each agent gets isolated state copy via deepcopy
- Collects results and performs majority voting
- Groups numeric values within 1% tolerance
- Returns consensus result or best available result
- Handles gracefully when agents crash or fail

Features:
- **Parallelism**: Uses `asyncio.gather()` for true parallel execution
- **Voting**: Groups by numeric proximity, picks most common result
- **Resilience**: Works even if 1-2/3 agents fail
- **Debugging**: Logs individual agent success/failure status

### 5a. Solver (`src/core/orchestrator/solver/solver.py`)
**Executes individual atomic steps** (async entry point, ~65 lines)

Main function: `async solve_step(step: Step, state: StateObject, sandbox: Optional[Sandbox])`
- Routes step execution based on operation type
- Delegates to execution and parsing modules
- Passes hot sandbox for reuse across steps
- Returns success status and result
- Now async for parallel swarm execution

#### 5b. Solver Execution (`src/core/orchestrator/solver/execution.py`)
**LLM-powered code generation and execution** (~520 lines - now with async versions)

Functions:
- `execute_with_llm()` - Single-output step execution (sync)
- `execute_with_llm_multi_output()` - Batch extraction execution (sync)
- `execute_with_llm_async()` - Single-output step execution (async - NEW)
- `execute_with_llm_multi_output_async()` - Batch extraction execution (async - NEW)
- Uses gpt-4.1-mini-2025-04-14 for code generation
- Supports up to 3 internal retry attempts if generation fails
- AsyncOpenAI client for true parallel execution

#### 5d. Solver Parsing (`src/core/orchestrator/solver/parsing.py`)
**Output processing and validation** (~200 lines)

Functions:
- `parse_output()` - Extracts value and unit from execution output
- `parse_multi_output()` - Handles batch extraction results
- `extract_result_from_text()` - Fallback text extraction
- `validate_result()` - Validates results meet semantic requirements
- `calculate_cost()` - Computes API token costs

#### 5e. Solver Prompts (`src/core/orchestrator/prompts/solver.py`)
**Prompt templates and builders** (~130 lines - updated with degree rules)

Constants:
- `SOLVER_SYSTEM_MESSAGE` - System message for single-output steps (includes degree rule: "For angles, ALWAYS output in DEGREES")
- `SOLVER_MULTI_OUTPUT_SYSTEM_MESSAGE` - System message for batch steps (includes degree rule)

Functions:
- `build_extract_prompt()` - Generates extraction prompts
- `build_calculate_prompt()` - Generates calculation prompts
- `build_convert_prompt()` - Generates unit conversion prompts

**New Feature: Degree Convention**
- Agents now output angles in degrees by default
- Prevents radian/degree confusion (e.g., Problem 4 in benchmarks)
- Rules: "Convert radian results to degrees before outputting: degrees = radians * (180 / π)"

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
import asyncio
from src.core.orchestrator.orchestrate import solve_problem

async def main():
    problem = "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?"
    result = await solve_problem(problem)

    if result['success']:
        print(f"Answer: {result['final_answer']} {result['final_unit']}")
        print(f"Plan steps: {len(result['plan'].steps)}")
        print(f"Total cost: ${result['total_cost']:.4f}")
        print(f"Execution time: {result['execution_time']:.2f}s")
    else:
        print(f"Error: {result['error']}")

# Run async
asyncio.run(main())
```

**Note**: solve_problem is now async to support K-Ahead Swarm's parallel execution. Use `await` when calling it, or `asyncio.run()` from sync code.

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

- **K-Ahead Swarm**: 3 parallel agents per step with majority voting - eliminates hallucinations and syntax crashes
- **Atomic Steps**: Each step performs exactly one operation for clarity and debuggability
- **Structured Planning**: Uses LLM to intelligently decompose problems
- **Physics Validation**: Physics Lawyer audits plans for conceptual errors before execution
- **Automatic Repair**: Revisor automatically fixes flagged plans while preserving dependencies
- **Async Execution**: Full async/await architecture for true parallel execution
- **Sandboxed Execution**: All code runs in E2B environment for security
- **State Tracking**: Maintains complete history of values and assumptions
- **Robust Error Recovery**: Swarm attempts up to 3 times (9 total agent attempts per step)
- **Degree Convention**: Angles output in degrees by default (no radian/degree confusion)
- **Numeric Comparison**: Robust extraction handles scientific notation, symbolic math, embedded numbers
- **Unit Handling**: Built-in support for unit conversion using Pint
- **Reproducibility**: Full plan and state are returned with results
- **Extended Timeout**: 5 minute timeout (300s) for complex problems

## Roadmap

- ✅ **Physics Lawyer & Revisor**: Conceptual error detection and automatic repair (COMPLETE)
- ✅ **K-Ahead Swarm**: Parallel execution with majority voting (COMPLETE)
- ✅ **Async Architecture**: Full async/await execution (COMPLETE)
- ✅ **Robust Numeric Comparison**: Handle all number formats (COMPLETE)
- ✅ **Degree Convention**: Angles in degrees by default (COMPLETE)
- ✅ **Extended Timeout**: 5 minute limit for complex problems (COMPLETE)
- ✅ **Benchmarking Suite**: Complete with SciBench evaluation (COMPLETE)
- Implement MDAP (multi-step degradation analysis process) for execution-time error reduction
- Domain-specific critics (thermodynamics, quantum mechanics, etc.)
- Interactive correction approval (allow user to review and approve fixes)
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
│           │   ├── solver.py            # Main entry point (ASYNC)
│           │   ├── swarm.py             # K-Ahead Swarm (NEW)
│           │   ├── execution.py         # LLM executors (with async versions)
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
- **Solving**: GPT-4.1-mini-2025-04-14 (via OpenAI) - optimized for code generation and variable naming (3 agents run in parallel)
- **Vision**: GPT-4o (via OpenAI) - advanced vision capabilities + tool calling for image analysis
- **Temperature**: 0.1 for all solvers - low randomness helps avoid degenerate code generation

### Performance Features

- **K-Ahead Swarm**: 3 agents execute in parallel per step via asyncio (true parallelism)
- **AsyncOpenAI**: Non-blocking API calls for parallel agent execution
- **Hot Sandbox**: Reuses E2B sandbox across all steps (~70% faster per step)
- **Direct OpenAI API**: Uses direct OpenAI connection when available (2-3s faster per step)
- **Batch Extraction**: Groups multiple variable extractions into single steps
- **Cost Tracking**: Real-time USD cost calculation for all API calls
- **Majority Voting**: Numeric proximity matching (1% tolerance) for consensus

### Expected Performance

- **Per-step solving time**: 6-10 seconds (swarm of 3 parallel agents, with hot sandbox)
  - Individual agent time: ~2-4 seconds per agent
  - Parallel execution: ~3x speedup when all agents work
  - Retry efficiency: Failed swarms retry without replanning
- **Per-image vision analysis**: 20-90 seconds with CV tools (simple extraction ≈20s, complex preprocessing ≈90s)
- **CV tool preprocessing**: 0-10 iterations max (typically 2-5 tools used)
- **Physics review time**: 3-8 seconds per audit pass
- **Overall pipeline**:
  - Simple problem: 1-2 minutes
  - Complex multi-step: 3-5 minutes
  - With image input: 4-8 minutes (includes vision analysis)
- **Robustness**: Swarm resilience means <1% failure rate even with occasional agent crashes

## Architecture & Design Decisions

### Why K-Ahead Swarm?

The K-Ahead Swarm pattern solves three critical problems:

1. **Syntax Errors**: LLMs occasionally generate invalid Python code (~5% of the time). Running 3 agents in parallel means if 1 crashes, 2 still succeed. Single-agent approach would fail entirely.

2. **Hallucinations**: LLMs can hallucinate variable names or values. For example, "use mass=5kg" when the problem says 10kg. Majority voting (3 agents) makes hallucination extremely unlikely - all 3 would need to hallucinate the SAME wrong value.

3. **Recovery Without Replanning**: Instead of replanning on error (expensive, wastes time), swarm retry is lightweight - just try again with the same step. No need to reanalyze the problem.

**Design Choice: Per-Step Swarm (not Per-Problem Swarm)**
- Per-problem would wastefully repeat planning 3 times (expensive)
- Per-step parallelism targets execution errors specifically
- Preserves state dependencies (critical for sequential solving)
- 3x cost per step vs 3x cost per entire problem

**Design Choice: Majority Voting by Numeric Proximity**
- Different agents may output slightly different values due to floating point
- Grouping within 1% tolerance captures legitimate variance
- Ensures consensus is real, not just luck

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

### Why 3-Tier Routing?

The routing system provides:
1. **Cost Optimization**: EASY problems get the cheap Flash model (~8-10x cost reduction)
2. **Speed**: EASY/MEDIUM problems skip expensive planning phase, ~2-3x faster
3. **Resource Efficiency**: Don't run orchestrator on trivial problems
4. **Flexibility**: Can be toggled off for backward compatibility
5. **Visibility**: Routing metadata helps understand problem distribution

**Design Choice: Fast Classification with gemini-3-flash-preview**
- Flash is 8-10x cheaper than Pro for routing decisions
- ~0.5-1 second overhead (negligible per problem)
- Cost: ~$0.0008 per problem (adds <1% to total cost)

## Usage & Configuration

### Enabling/Disabling Routing

The routing system is **enabled by default**. To use it:

```bash
# Default: Routing ENABLED (3-tier system)
python -m src.benchmarking.cli run --config benchmark_configs/scibench_fast.yaml

# Disable routing: Use legacy dispatch logic
USE_ROUTING=false python -m src.benchmarking.cli run --config benchmark_configs/scibench_fast.yaml
```

### Environment Variables

**Routing Control:**
- `USE_ROUTING=true` (default) - Enable intelligent 3-tier routing
- `USE_ROUTING=false` - Disable routing, use legacy `USE_SINGLE_AGENT` logic

**Legacy Mode (when `USE_ROUTING=false`):**
- `USE_SINGLE_AGENT=true` - Use single-agent solver
- `USE_SINGLE_AGENT=false` (default) - Use multi-agent orchestrator

**Model Selection:**
- `GOOGLE_API_KEY` - Required for all Gemini models (routing, planner, etc.)
- `OPENAI_API_KEY` - Optional for vision/GPT-4o functionality
- `OPEN_ROUTER_KEY` - Fallback for OpenAI models if no OPENAI_API_KEY

**Physics Validation:**
- `SKIP_PHYSICS_REVIEW=true` - Skip physics lawyer validation (faster, less safe)

### Example: Running with Routing

```bash
# Set up environment
export GOOGLE_API_KEY="your-google-api-key"

# Run benchmark with routing enabled
python -m src.benchmarking.cli run --config benchmark_configs/scibench_fast.yaml

# Watch the routing output
# Example output:
# 🚦 Router: EASY (confidence: 0.92)
#    Reasoning: Single definition and one-step calculation
#    Time: 0.45s | Cost: $0.000082
#    → Using single-agent solver with gemini-3-flash-preview
#
# 🚦 Router: MEDIUM (confidence: 0.78)
#    Reasoning: Standard textbook problem with 3 clear steps
#    Time: 0.52s | Cost: $0.000095
#    → Using single-agent solver with gemini-3-pro-preview
#
# 🚦 Router: HARD (confidence: 0.85)
#    Reasoning: Multi-page derivation requiring strategic planning
#    Time: 0.61s | Cost: $0.000103
#    → Using multi-agent orchestrator (Planner + Swarm)
```

### Analyzing Routing Results

The routing system adds metadata to results for analysis:

```python
from src.benchmarking.runner.problem_executor import ProblemExecutor

executor = ProblemExecutor(timeout_seconds=300)
result = executor.execute(problem)

# Access routing information
print(f"Tier: {result.routing_tier}")          # "EASY", "MEDIUM", or "HARD"
print(f"Confidence: {result.routing_confidence}")  # 0.0-1.0
print(f"Cost: ${result.routing_cost:.6f}")    # Routing overhead cost
print(f"Time: {result.routing_time:.2f}s")    # Routing classification time
print(f"Reasoning: {result.routing_reasoning}")    # Why this tier was selected

# Analyze cost and time breakdown
print(f"\n--- Timing ---")
print(f"Router time: {result.routing_time:.2f}s")
print(f"Solver time: {result.total_time - result.routing_time:.2f}s")
print(f"Total time: {result.total_time:.2f}s")

print(f"\n--- Cost Breakdown ---")
print(f"Router cost: ${result.routing_cost:.6f}")
print(f"Solver cost: ${result.total_cost - result.routing_cost:.6f}")
print(f"Total cost: ${result.total_cost:.6f}")
```

### Troubleshooting Routing

**Router Classification Seems Wrong:**
- The router uses a 120-line prompt with detailed tier definitions
- Confidence score indicates how sure the router is (0.0-1.0)
- For debugging, add custom scoring logic or adjust prompt in `src/core/router.py`

**Router Failing (Defaulting to MEDIUM):**
- Check that `GOOGLE_API_KEY` is set
- Verify gemini-3-flash-preview model is available in your region
- Router falls back to MEDIUM tier on any error (safe choice)

**Need Legacy Behavior:**
- Set `USE_ROUTING=false` to use old dispatch logic
- Router is only called if `USE_ROUTING=true`

### Single-Agent Solver Model Parameter

The single-agent solver now accepts a `model` parameter:

```python
from src.core.single_agent.solver import solve_problem

# EASY tier: Use Flash (fast & cheap)
result = solve_problem(
    problem="What is F=ma?",
    model="gemini-3-flash-preview"
)

# MEDIUM tier: Use Pro (good reasoning)
result = solve_problem(
    problem="A ball is thrown at 20 m/s at 45°. Find max height.",
    model="gemini-3-pro-preview"  # Default
)

# Or use directly without routing
result = solve_problem(problem=your_problem)  # Uses default gemini-3-pro-preview
```

### Pricing Configuration

Model pricing is centralized in `src/core/orchestrator/config/pricing.py`:

```python
MODEL_PRICING = {
    "gemini-3-flash-preview": {
        "input": 0.5,      # $0.50 per 1M input tokens
        "output": 3.0,     # $3.00 per 1M output tokens
    },
    "gemini-3-pro-preview": {
        "input": 2.0,      # $2.00 per 1M input tokens
        "output": 12.0,    # $12.00 per 1M output tokens
    },
    # ... more models
}
```

To add a new model or update pricing, edit this file.

## Notes

- The system is optimized for physics and mathematics problems
- All mathematical operations use SI units or standard physics units
- Code generation and execution happen dynamically based on problem requirements
- Refactored for code clarity with zero performance impact (Python imports are cached)
