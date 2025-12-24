from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union
from enum import Enum

class Variable(BaseModel):
    name: str
    description: str
    expected_unit: str
    value: Optional[Union[float, str]] = None  # float for numeric, str for symbolic
    unit: Optional[str] = None
    source_step: Optional[int] = None
    is_symbolic: bool = False  # True if this is a symbolic variable
    extraction_hint: Optional[str] = None  # e.g., "table row 4, column 2" or "at (x=4, y=4)"

class StateObject(BaseModel):
    problem_text: str
    domain: str
    variables: Dict[str, Variable]
    assumptions: List[str] = Field(default_factory=list)
    problem_context: Optional[dict] = None  # Store image_path and other runtime context

class OperationType(str, Enum):
    EXTRACT = "extract"
    CALCULATE = "calculate"
    CONVERT = "convert"
    OBSERVE = "observe"  # Query vision model for visual properties

class Step(BaseModel):
    step_id: int
    operation: OperationType
    description: str
    inputs: List[str]
    output: Optional[str] = None  # For backward compatibility - single output
    outputs: Optional[List[str]] = None  # Multiple outputs (for batched extractions)
    formula: Optional[str] = None
    expected_unit: Optional[str] = None  # For single output steps
    expected_units: Optional[Dict[str, str]] = None  # For multi-output steps: {var_name: unit}
    justification: str = Field(..., max_length=150)
    is_symbolic: bool = False  # True if this step involves symbolic variables

    def get_outputs(self) -> List[str]:
        """Get outputs as a list, handling both single and multiple outputs."""
        if self.outputs:
            return self.outputs
        elif self.output:
            return [self.output]
        return []

    def get_unit(self, var_name: str) -> str:
        """Get the expected unit for a specific variable."""
        if self.expected_units and var_name in self.expected_units:
            return self.expected_units[var_name]
        return self.expected_unit or ""

class Plan(BaseModel):
    steps: List[Step]
    final_output: str
    approach: str