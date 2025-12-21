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

class StateObject(BaseModel):
    problem_text: str
    domain: str
    variables: Dict[str, Variable]
    assumptions: List[str] = Field(default_factory=list)

class OperationType(str, Enum):
    EXTRACT = "extract"
    CALCULATE = "calculate"
    CONVERT = "convert"

class Step(BaseModel):
    step_id: int
    operation: OperationType
    description: str
    inputs: List[str]
    output: str
    formula: Optional[str] = None
    expected_unit: str
    justification: str = Field(..., max_length=150)
    is_symbolic: bool = False  # True if this step involves symbolic variables

class Plan(BaseModel):
    steps: List[Step]
    final_output: str
    approach: str