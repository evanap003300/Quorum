# Physics Lawyer & Revisor Architecture

## Overview

The **Physics Lawyer (Critic)** and **Revisor (Architect)** form a two-stage validation loop that catches conceptual physics errors BEFORE execution. This prevents wasted computation on fundamentally flawed plans.

### Architecture Flow

```
[Planner] 
    ↓
[Draft Plan] 
    ↓
[Physics Lawyer Audit]
    ↓
    ├─ APPROVED → [Execute Plan] ✓
    │
    └─ REJECTED → [Physics Lawyer Critiques]
         ↓
         [Revisor Repairs Plan]
         ↓
         [Re-audit] (max 2 retries)
         ↓
         ├─ APPROVED → [Execute Plan] ✓
         │
         └─ Max retries reached → [Execute with Warning] ⚠️
```

## Components

### 1. Physics Lawyer (Critic)
**File:** `src/core/orchestrator/planner/critics/physics_lawyer.py`

**Purpose:** Audits problem-solving plans for physics violations BEFORE code execution.

**Key Features:**
- Checks for reference frame errors (e.g., using absolute velocity when relative velocity is needed)
- Validates variable mass systems (e.g., F = dp/dt for rockets)
- Ensures conservation laws are applied correctly
- Detects small-angle approximation violations
- Validates unit consistency
- Checks physical constraints (velocities < c, etc.)
- Verifies step dependencies

**Usage:**
```python
from planner.critics.physics_lawyer import audit_plan

audit_result = audit_plan(problem_text, plan_object)

if audit_result.is_approved:
    print("✅ Plan is physically sound")
else:
    for critique in audit_result.critiques:
        print(f"Step {critique['step_index']}: {critique['error']}")
        print(f"  → Fix: {critique['correction']}")
```

**Returns:** `AuditResult` with:
- `status`: "APPROVED" or "REJECTED"
- `reasoning`: Brief explanation
- `critiques`: List of specific errors found
  - `step_index`: Which step has the error
  - `severity`: "BLOCKING" or "WARNING"
  - `category`: Which audit checklist item (Reference Frames, Variable Mass, etc.)
  - `error`: Specific conceptual error
  - `correction`: Exact fix needed
  - `affected_steps`: Downstream steps that depend on this error

### 2. Revisor (Architect)
**File:** `src/core/orchestrator/planner/revisor.py`

**Purpose:** Surgically repairs plans flagged by the Physics Lawyer while preserving dependencies.

**Key Features:**
- Locates flagged steps by index
- Rewrites step logic to implement corrections
- Preserves downstream dependencies (variable names, units, outputs)
- Can insert new intermediate steps if needed
- Maintains step numbering/ordering
- Returns corrected plan in same JSON schema

**Usage:**
```python
from planner.revisor import revise_plan

revised_plan = revise_plan(
    problem=problem_text,
    original_plan=plan_object,
    critiques=audit_result.critiques
)
```

**Returns:** New `Plan` object with corrections applied

### 3. Review Loop Integration
**Location:** `src/core/orchestrator/orchestrate.py` - `_review_and_revise_plan()`

**Flow:**
1. After planning stage creates a draft plan
2. Physics Lawyer audits the plan
3. If approved → proceed to execution
4. If rejected:
   - Show specific critiques to user
   - Call Revisor to fix flagged steps
   - Re-audit revised plan (max 2 retries)
   - If still invalid after max retries → proceed with warning

**Key Parameters:**
- `max_revisions`: Number of retry attempts (default: 2)
- Each revision is one complete audit→repair cycle

## Prompts

### Physics Lawyer Prompt
**File:** `src/core/orchestrator/prompts/critics.py`

**System Message Content:**
- Audit checklist with 7 major categories:
  1. Reference Frames & Relative Motion
  2. Variable Mass Systems
  3. Conservation Laws
  4. Small Angle/Perturbation Approximations
  5. Unit Consistency
  6. Physical Constraints
  7. Step Dependencies

- JSON output schema for structured critiques
- Instructions to flag only real physics errors (not implementation details)

### Revisor Prompt
**File:** `src/core/orchestrator/prompts/revisor.py`

**System Message Content:**
- Instructions for plan repair strategy
- Rules for preserving dependencies
- Guidance on variable naming consistency
- When to add intermediate steps
- Output format requirements

## Integration Points

### In `orchestrate.py`:

```python
# After planning, before execution:
print("PHYSICS REVIEW")
plan_obj = _review_and_revise_plan(problem, plan_obj, max_revisions=2)

# Then proceed to execution:
print("EXECUTION")
# ... execute steps ...
```

### Dependencies:
- Imports `audit_plan` from `planner.critics.physics_lawyer`
- Imports `revise_plan` from `planner.revisor`
- Calls `_review_and_revise_plan()` with problem text and plan object

## Error Handling

### If Physics Lawyer audit fails:
- Error message printed to user
- Continues with warning (best-effort approach)
- Execution proceeds with flagged plan

### If Revisor revision fails:
- Error logged
- Reverts to original plan
- Continues execution with warning

### Max revisions reached:
- Warning displayed to user
- Continues with best-effort plan
- All subsequent execution proceeds normally

## Example Output

### Approved Plan:
```
⚖️ Physics Lawyer reviewing plan (Pass 1/2)...
✅ Plan approved by Physics Lawyer.
   Reasoning: All steps maintain physical consistency and conservation laws.
```

### Rejected Plan (with Revisor):
```
⚖️ Physics Lawyer reviewing plan (Pass 1/2)...
❌ Physics Lawyer found 1 issue(s):
   🔴 Step 3: Plan calculates invariant using absolute velocity V, but the wall is moving.
      → Fix: Use relative velocity (v_ball - V_wall) for the invariant J = p*q.

🔧 Revisor is patching the plan...
✓ Plan revised. Retrying audit...

⚖️ Physics Lawyer reviewing plan (Pass 2/2)...
✅ Plan approved by Physics Lawyer.
```

## Design Decisions

### Why Two-Stage Review?
1. **Physics Lawyer** catches conceptual errors early (before expensive code execution)
2. **Revisor** automatically repairs fixable issues (reduces manual intervention)
3. **Feedback loop** ensures corrections are themselves sound

### Why Max 2 Revisions?
- Balances correction quality vs computational cost
- Most errors are caught in first revision
- Prevents infinite loops

### Why Not Just Use Planner Again?
- Physics Lawyer is specifically trained to audit (adversarial role)
- Revisor understands dependency chains better than regenerating from scratch
- Smaller LLM calls = faster correction cycle

## Testing the Review Loop

To test the critic/revisor system end-to-end:

```python
from orchestrate import solve_problem

# Problem with a subtle physics error
problem = "..."

result = solve_problem(problem)

# Will show:
# 1. Initial plan creation
# 2. Physics Lawyer audit (may find issues)
# 3. Revisor fixes (if needed)
# 4. Execution with approved plan
```

## Future Enhancements

- **More Critics:** Domain-specific critics (mechanics, thermodynamics, etc.)
- **Confidence Scoring:** Return confidence levels for audit results
- **Suggestion Ranking:** Prioritize most impactful fixes when multiple issues found
- **Interactive Mode:** Allow user to approve/reject revisor's proposed fixes
- **Learning Loop:** Track which errors are common and improve detection

## Files Created

```
src/core/orchestrator/
├── planner/
│   ├── critics/
│   │   ├── __init__.py
│   │   └── physics_lawyer.py      # Audit logic
│   └── revisor.py                  # Repair logic
│
└── prompts/
    ├── critics.py                  # Physics Lawyer prompts
    └── revisor.py                  # Revisor prompts
```

## Integration with Existing System

- **No breaking changes** to existing API
- `solve_problem()` function signature unchanged
- Review loop is transparent to external callers
- Return value structure unchanged
- Can be disabled by skipping the review step (future enhancement)
