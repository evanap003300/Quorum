from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum

class Variable(BaseModel):
    name: str
    description: str
    expected_unit: str
    value: Optional[float] = None
    unit: Optional[str] = None
    source_step: Optional[int] = None

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

class Plan(BaseModel):
    steps: List[Step]
    final_output: str
    approach: str