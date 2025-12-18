Implementation Plan: Planner System for Quorum                                    
                                                                                       
     Overview                                                                          
                                                                                       
     Implement a production-ready Planner system (planner.py) that replaces the        
     simple plan.py function with a robust, schema-validated planning architecture.    
      The planner will generate atomic, structured plans (JSON only) with              
     critique-refine loops and strict validation.                                      
                                                                                       
     Core Principle: Parse natural language problems → structured State Object →       
     atomic Plan (no numeric computation, only symbolic operations) → validated by     
     Critics → approved by Gatekeeper → executed by MDAP solver.                       
                                                                                       
     ---                                                                               
     Architecture Summary                                                              
                                                                                       
     User Decisions                                                                    
                                                                                       
     - Self-contained planner.py (replaces plan.py, not a wrapper)                     
     - Planner only with interfaces (abstract base classes for Critics/Gatekeeper;     
     implement later)                                                                  
     - Separate schemas.py for all Pydantic models                                     
     - Pydantic v2 with strict mode for type safety                                    
                                                                                       
     Key Components                                                                    
                                                                                       
     1. schemas.py: Pydantic v2 models with strict validation                          
       - StateObject: Master state with problem, knowns, assumptions, audit log        
       - Plan: Versioned plan with atomic steps and composition tracking               
       - AtomicStep: Single-operation step                                             
     (algebraic|symbolic|matrix|ode|substitute)                                        
       - CritiqueUnit: Interface for critic feedback                                   
     2. planner.py: Main orchestration class                                           
       - LLM integration (OpenRouter client matching existing patterns)                
       - Retry logic (max 2 retries on schema failures)                                
       - Critique-refine loop (up to 2 iterations)                                     
       - Temperature 0.1 for deterministic planning                                    
     3. interfaces.py: Abstract base classes                                           
       - BaseCritic: Protocol for plan critique                                        
       - BaseGatekeeper: Protocol for plan approval                                    
       - SolverProtocol: Future integration point for MDAP                             
     4. validators.py: Business logic for validation                                   
       - Atomicity enforcement (detect compound operations via regex + heuristics)     
       - Plan structure validation (dependencies, unique IDs, unit consistency)        
     5. prompts.py: LLM prompt engineering                                             
       - System prompt emphasizing atomicity and JSON output                           
       - User prompt template with problem context                                     
       - Few-shot examples of valid atomic plans                                       
     6. exceptions.py: Custom exception hierarchy                                      
       - PlanningError, SchemaValidationError, LLMError, AtomicityViolationError       
                                                                                       
     ---                                                                               
     Critical Files to Implement (Priority Order)                                      
                                                                                       
     Phase 1: Foundation (Implement First)                                             
                                                                                       
     1. schemas.py (HIGHEST PRIORITY)                                                  
     - Path: src/core/orchestrator/planner/schemas.py                                  
     - Defines: AtomicStep, Plan, StateObject, CritiqueUnit                            
     - Key features:                                                                   
       - Strict mode (extra='forbid') to prevent silent failures                       
       - Justification length validator (≤50 tokens = ~200 chars)                      
       - Audit log in StateObject for traceability                                     
       - UUID-based step IDs for global uniqueness                                     
     - Why first: Everything depends on these schemas                                  
                                                                                       
     2. validators.py                                                                  
     - Path: src/core/orchestrator/planner/validators.py                               
     - Functions:                                                                      
       - validate_step_atomicity(step): Detect compound operations                     
       - validate_plan_structure(plan): Check dependencies, uniqueness                 
     - Patterns to detect:                                                             
       - Multiple verbs (solve AND substitute)                                         
       - Sequence indicators (first, then, finally)                                    
       - Conjunction patterns (and, then, comma-separated actions)                     
     - Why second: Schemas reference these validators in field validators              
                                                                                       
     3. exceptions.py                                                                  
     - Path: src/core/orchestrator/planner/exceptions.py                               
     - Define: PlanningError, SchemaValidationError, LLMError,                         
     AtomicityViolationError, CritiqueError                                            
     - Why third: Used throughout planner.py for error handling                        
                                                                                       
     Phase 2: LLM Integration                                                          
                                                                                       
     4. prompts.py                                                                     
     - Path: src/core/orchestrator/planner/prompts.py                                  
     - Functions:                                                                      
       - get_planning_system_prompt(): Fixed system message                            
       - get_planning_user_prompt(state): Dynamic user message                         
     - Key instructions in system prompt:                                              
       - Each step must be ATOMIC (single operation only)                              
       - Output JSON only, no freeform text                                            
       - Justifications ≤50 tokens                                                     
       - Flag missing data with assumption_needed                                      
     - Include few-shot examples of atomic vs compound steps                           
     - Why fourth: Determines quality of LLM outputs                                   
                                                                                       
     5. interfaces.py                                                                  
     - Path: src/core/orchestrator/planner/interfaces.py                               
     - Abstract classes:                                                               
       - BaseCritic: critique(plan) -> list[CritiqueUnit]                              
       - BaseGatekeeper: should_refine(critiques) -> tuple[bool, list]                 
       - SolverProtocol: Future MDAP integration                                       
     - Why fifth: Defines contracts for extensibility                                  
                                                                                       
     Phase 3: Core Orchestration                                                       
                                                                                       
     6. planner.py (MAIN IMPLEMENTATION)                                               
     - Path: src/core/orchestrator/planner/planner.py                                  
     - Class: Planner                                                                  
     - Key methods:                                                                    
       - __init__(model, temperature, max_retries, critics, gatekeeper)                
       - create_plan(problem_text, domain) -> StateObject                              
       - _generate_plan_with_retries(state) -> Plan (private)                          
       - _critique_refine_loop(state, max_iterations=2) -> StateObject (private)       
       - _refine_plan(state, critiques) -> Plan (private)                              
     - Integration:                                                                    
       - Uses existing OpenRouter pattern from plan.py/solver.py                       
       - Force JSON output: response_format={"type": "json_object"}                    
       - Model: "openai/gpt-4o" (strong model for planning)                            
       - Temperature: 0.1 (low but not zero)                                           
     - Retry policy:                                                                   
       - Max 2 retries on schema validation failure                                    
       - Include error message in retry prompt to guide LLM                            
       - Log all attempts for debugging                                                
     - Why last: Orchestrates all other components                                     
                                                                                       
     ---                                                                               
     Implementation Steps                                                              
                                                                                       
     Step 1: Create schemas.py                                                         
                                                                                       
     1. Set up Pydantic v2 with ConfigDict(strict=True, extra='forbid')                
     2. Implement AtomicStep with:                                                     
       - Type enum: algebraic|symbolic|matrix|ode|substitute                           
       - Justification length validator (≤200 chars = ~50 tokens)                      
       - Required fields: type, description, inputs, output                            
       - Optional: expected_units, tolerance                                           
     3. Implement Plan with:                                                           
       - plan_version (int, starts at 1)                                               
       - variables (dict of variable descriptions)                                     
       - assumptions (list of strings)                                                 
       - steps (list of AtomicStep, min_length=1)                                      
       - composition (dependency tracking)                                             
       - metadata (model, timestamp)                                                   
     4. Implement StateObject with:                                                    
       - state_id (UUID)                                                               
       - problem_text, domain, goal                                                    
       - knowns, unknowns, assumptions, constraints                                    
       - plan (optional Plan)                                                          
       - audit_log (list of dicts)                                                     
       - log_event() helper method                                                     
     5. Implement CritiqueUnit interface                                               
     6. Write unit tests for all schemas                                               
                                                                                       
     Step 2: Create validators.py                                                      
                                                                                       
     1. Implement validate_step_atomicity(step):                                       
       - Regex patterns for compound operations (and, then, comma sequences)           
       - Heuristic: count action verbs (solve, substitute, differentiate, etc.)        
       - Raise AtomicityViolationError if compound detected                            
     2. Implement validate_plan_structure(plan):                                       
       - Check step IDs are unique                                                     
       - Check output variables are unique                                             
       - Check dependencies reference valid variables                                  
       - Raise ValueError with clear messages                                          
     3. Write comprehensive unit tests (valid steps, compound steps, edge cases)       
                                                                                       
     Step 3: Create exceptions.py                                                      
                                                                                       
     1. Define exception hierarchy:                                                    
       - PlanningError (base)                                                          
       - SchemaValidationError (LLM output doesn't match schema)                       
       - LLMError (API call fails)                                                     
       - AtomicityViolationError (step violates atomicity)                             
       - CritiqueError (critique-related failures)                                     
     2. Add detailed error messages with remediation hints                             
                                                                                       
     Step 4: Create prompts.py                                                         
                                                                                       
     1. Write get_planning_system_prompt():                                            
       - Expert planner persona                                                        
       - Emphasize atomicity (each step = one operation)                               
       - JSON-only output requirement                                                  
       - Short justifications (≤50 tokens)                                             
       - Flag missing data with assumption_needed                                      
       - Include examples of atomic vs compound steps                                  
     2. Write get_planning_user_prompt(state):                                         
       - Format: problem text + domain                                                 
       - Include expected JSON schema structure                                        
       - Remind about atomicity requirement                                            
     3. Add few-shot examples (e.g., kinematics problem with 2 atomic steps)           
     4. Manual testing: run prompts through LLM, verify JSON quality                   
                                                                                       
     Step 5: Create interfaces.py                                                      
                                                                                       
     1. Define BaseCritic abstract class:                                              
       - Property: critic_id (string identifier)                                       
       - Method: critique(plan: Plan) -> list[CritiqueUnit]                            
       - Docstrings explaining contract                                                
     2. Define BaseGatekeeper abstract class:                                          
       - Method: should_refine(critiques) -> tuple[bool, list[CritiqueUnit]]           
       - Returns: (should_refine: bool, blocking_issues: list)                         
     3. Define SolverProtocol for future integration:                                  
       - Methods: execute_step(), execute_plan()                                       
     4. Add example implementations in docstrings                                      
                                                                                       
     Step 6: Create planner.py                                                         
                                                                                       
     1. Setup (lines 1-30):                                                            
       - Imports: os, json, logging, dotenv, openai, schemas, interfaces,              
     validators, prompts, exceptions                                                   
       - Initialize logger                                                             
       - Load environment variables                                                    
     2. Planner.init() (lines 32-50):                                                  
       - Parameters: model, temperature, max_retries, critics, gatekeeper              
       - Create OpenAI client (match pattern from plan.py):                            
       self.client = OpenAI(                                                           
         base_url="https://openrouter.ai/api/v1",                                      
         api_key=os.getenv("OPEN_ROUTER_KEY")                                          
     )                                                                                 
       - Store critics and gatekeeper (optional, default to empty/None)                
       - Log initialization                                                            
     3. create_plan() (lines 52-75):                                                   
       - Main entry point                                                              
       - Create initial StateObject from problem_text and domain                       
       - Call _generate_plan_with_retries()                                            
       - If critics/gatekeeper present, call _critique_refine_loop()                   
       - Return StateObject with validated Plan                                        
     4. _generate_plan_with_retries() (lines 77-125):                                  
       - Loop: max_retries + 1 attempts                                                
       - Build messages: system prompt + user prompt                                   
       - Call LLM with:                                                                
           - model (default: openai/gpt-4o)                                            
         - temperature (default: 0.1)                                                  
         - response_format: {"type": "json_object"}                                    
       - Parse JSON response                                                           
       - Validate with Pydantic (Plan(**plan_data))                                    
       - Additional validation: validate_plan_structure(plan)                          
       - On failure: log error, retry                                                  
       - After max retries: raise PlanningError                                        
     5. _critique_refine_loop() (lines 127-160):                                       
       - Max 2 iterations (per spec)                                                   
       - Collect critiques from all critics                                            
       - If no critiques, approve and break                                            
       - Call gatekeeper.should_refine(critiques)                                      
       - If should_refine, call _refine_plan()                                         
       - Update state.plan, increment version                                          
       - Log all critique rounds                                                       
     6. _refine_plan() (lines 162-190):                                                
       - Build messages with critique context:                                         
           - Original system/user prompt                                               
         - Previous plan (as assistant message)                                        
         - Critiques (as user feedback)                                                
       - Call LLM with temperature=0.0 (deterministic refinement)                      
       - Parse and validate refined plan                                               
       - Increment plan_version                                                        
       - Return refined Plan                                                           
     7. Error handling:                                                                
       - Catch json.JSONDecodeError → SchemaValidationError                            
       - Catch ValueError (Pydantic) → SchemaValidationError                           
       - Catch generic Exception → LLMError                                            
       - Log all errors with context                                                   
     8. Write integration test (test end-to-end with real LLM call)                    
                                                                                       
     ---                                                                               
     Testing Strategy                                                                  
                                                                                       
     Unit Tests (tests/unit/)                                                          
                                                                                       
     test_schemas.py:                                                                  
     - Test justification length validation (should reject >200 chars)                 
     - Test strict mode (should reject extra fields)                                   
     - Test StateObject.log_event() functionality                                      
     - Test Plan requires at least one step                                            
     - Test AtomicStep type enum validation                                            
                                                                                       
     test_validators.py:                                                               
     - Test valid atomic steps pass validation                                         
     - Test compound operations are detected:                                          
       - "solve and substitute" → AtomicityViolationError                              
       - "differentiate then integrate" → AtomicityViolationError                      
       - Multiple verbs → AtomicityViolationError                                      
     - Test plan structure validation:                                                 
       - Duplicate step IDs → ValueError                                               
       - Undefined input variables → ValueError                                        
       - Circular dependencies → ValueError                                            
                                                                                       
     test_prompts.py:                                                                  
     - Test prompt generation with sample StateObject                                  
     - Test prompt includes atomicity instructions                                     
     - Test prompt includes JSON schema                                                
                                                                                       
     Integration Tests (tests/integration/)                                            
                                                                                       
     test_planner.py:                                                                  
     - Test planner initialization with defaults                                       
     - Test create_plan() with simple problem (mock LLM)                               
     - Test retry logic on invalid JSON (mock multiple LLM calls)                      
     - Test critique-refine loop (mock critics + gatekeeper)                           
     - Test end-to-end with real LLM call (marked as @pytest.mark.integration)         
                                                                                       
     Example test problems:                                                            
     1. Simple kinematics: "A car accelerates from 0 to 60 mph in 5 seconds. What      
     is its acceleration?"                                                             
       - Expected: 2 atomic steps (unit conversion, kinematic formula)                 
     2. Compound operation (should be rejected): "Solve for x then substitute into     
     equation 2"                                                                       
     3. Missing info: "Calculate the force on an object"                               
       - Expected: assumption_needed flag                                              
                                                                                       
     ---                                                                               
     Integration with Existing Code                                                    
                                                                                       
     Match Existing Patterns                                                           
                                                                                       
     From plan.py and solver.py:                                                       
     # Existing pattern (plan.py lines 5-12)                                           
     load_dotenv()                                                                     
     client = OpenAI(                                                                  
         base_url="https://openrouter.ai/api/v1",                                      
         api_key=os.getenv("OPEN_ROUTER_KEY")                                          
     )                                                                                 
     → Planner will use identical client initialization                                
                                                                                       
     Connection Points                                                                 
                                                                                       
     1. orchestrate.py (currently TODO):                                               
       - Will import: from .planner.planner import Planner                             
       - Will call: planner.create_plan(problem_text) → StateObject                    
       - Then pass to solver.py for execution                                          
     2. solver.py (existing):                                                          
       - Will receive: StateObject with validated Plan                                 
       - Will execute: each AtomicStep via MDAP voting                                 
       - Will use: k=3 voting from voting_pool.py                                      
     3. python_interpreter-e2b/main.py (existing):                                     
       - Will receive: code from solver agents                                         
       - Will execute: in sandbox with numpy, sympy, pint                              
       - Will return: validated results                                                
                                                                                       
     Migration from plan.py                                                            
                                                                                       
     Old usage:                                                                        
     from src.core.orchestrator.planner.plan import plan                               
     result = plan("What is 9 + 10")  # Returns string                                 
                                                                                       
     New usage:                                                                        
     from src.core.orchestrator.planner.planner import Planner                         
     planner = Planner()                                                               
     state = planner.create_plan("What is 9 + 10", domain="mathematics")               
     # Access structured data:                                                         
     # - state.plan.steps (list of AtomicStep)                                         
     # - state.plan.assumptions (list of strings)                                      
     # - state.audit_log (full history)                                                
                                                                                       
     Deprecation plan:                                                                 
     - Keep plan.py with deprecation notice for reference                              
     - Point to planner.py in comments                                                 
     - Remove plan.py after orchestrate.py is updated                                  
                                                                                       
     ---                                                                               
     Key Design Constraints (Must Enforce)                                             
                                                                                       
     1. No numeric computation in Planner                                              
       - Planner generates symbolic operations only                                    
       - Actual computation happens in solver + e2b sandbox                            
     2. Atomicity is sacred                                                            
       - Each step = exactly one operation                                             
       - Validators must catch compound operations                                     
       - Better to over-decompose than under-decompose                                 
     3. No silent state mutation                                                       
       - Only Cleaner/Validator (future) can commit to knowns                          
       - Planner only proposes operations                                              
     4. Structured data only                                                           
       - No freeform text in programmatic fields                                       
       - Justifications are short, structured strings                                  
     5. Traceability                                                                   
       - Audit log tracks every operation                                              
       - UUIDs for reproducibility                                                     
       - Metadata includes model, timestamp                                            
     6. Fail fast with clear errors                                                    
       - Schema validation failures are loud                                           
       - Error messages include remediation hints                                      
       - Max 2 retries, then escalate                                                  
                                                                                       
     ---                                                                               
     Dependencies                                                                      
                                                                                       
     Add to requirements.txt:                                                          
     # Core (existing)                                                                 
     python>=3.12                                                                      
     python-dotenv>=1.0.0                                                              
     openai>=1.0.0                                                                     
                                                                                       
     # NEW: Schema validation                                                          
     pydantic>=2.5.0                                                                   
                                                                                       
     # Development (existing)                                                          
     pytest>=7.4.0                                                                     
     pytest-cov>=4.1.0                                                                 
     pytest-mock>=3.12.0                                                               
                                                                                       
     # Optional: Type checking                                                         
     mypy>=1.7.0                                                                       
                                                                                       
     Environment variables (.env):                                                     
     OPEN_ROUTER_KEY=sk-or-v1-...  # Existing                                          
     LOG_LEVEL=INFO  # Optional                                                        
     PLANNER_MODEL=openai/gpt-4o  # Optional override                                  
     PLANNER_TEMPERATURE=0.1  # Optional override                                      
                                                                                       
     ---                                                                               
     Success Criteria                                                                  
                                                                                       
     Must Have (MVP)                                                                   
                                                                                       
     - Schemas validate correctly with Pydantic v2 strict mode                         
     - Atomicity validator catches compound operations                                 
     - Planner generates valid JSON plans from natural language                        
     - Retry logic works (max 2 retries on schema failures)                            
     - Unit test coverage >90% for schemas and validators                              
     - Integration test passes with real LLM call                                      
     - Abstract interfaces defined for Critics/Gatekeeper                              
                                                                                       
     Should Have (Polish)                                                              
                                                                                       
     - Comprehensive error messages with remediation hints                             
     - Audit log captures all planning operations                                      
     - Example critic and gatekeeper implementations in tests/                         
     - Documentation: docstrings for all public methods                                
     - Type hints verified with mypy                                                   
                                                                                       
     Nice to Have (Future)                                                             
                                                                                       
     - Async support for parallel critic calls                                         
     - Plan caching for identical problems                                             
     - Visualization of plan dependencies                                              
     - Telemetry dashboard for retry rates and error types                             
                                                                                       
     ---                                                                               
     Timeline Estimate                                                                 
                                                                                       
     Phase 1 (Foundation): 2-3 hours                                                   
     - schemas.py + validators.py + exceptions.py + unit tests                         
                                                                                       
     Phase 2 (LLM Integration): 1-2 hours                                              
     - prompts.py + interfaces.py + manual testing                                     
                                                                                       
     Phase 3 (Orchestration): 3-4 hours                                                
     - planner.py + integration tests + debugging                                      
                                                                                       
     Phase 4 (Polish): 1-2 hours                                                       
     - Documentation + examples + cleanup                                              
                                                                                       
     Total: ~7-11 hours of focused implementation                                      
                                                                                       
     ---                                                                               
     Risk Mitigation                                                                   
                                                                                       
     Risk 1: LLM doesn't follow JSON schema                                            
                                                                                       
     Mitigation:                                                                       
     - Use response_format={"type": "json_object"} (forces JSON)                       
     - Include schema structure in prompt                                              
     - Validate with Pydantic and retry up to 2x                                       
                                                                                       
     Risk 2: Atomicity validation too strict                                           
                                                                                       
     Mitigation:                                                                       
     - Start with conservative patterns                                                
     - Collect false positives during testing                                          
     - Tune regex patterns based on real examples                                      
                                                                                       
     Risk 3: Justification length hard to enforce                                      
                                                                                       
     Mitigation:                                                                       
     - Use character count (~4 chars/token) as proxy                                   
     - Pydantic validator rejects at parse time                                        
     - Prompt includes explicit ≤50 token instruction                                  
                                                                                       
     Risk 4: Plan agent and Explore agent suggestions diverge                          
                                                                                       
     Mitigation:                                                                       
     - This plan synthesizes both agent recommendations                                
     - User decisions lock in key choices (Pydantic v2, self-contained, etc.)          
     - Implementation follows detailed Plan agent design                               
                                                                                       
     ---                                                                               
     Critical Files Summary                                                            
                                                                                       
     Must create (in order):                                                           
     1. src/core/orchestrator/planner/schemas.py - Foundation schemas                  
     2. src/core/orchestrator/planner/validators.py - Atomicity enforcement            
     3. src/core/orchestrator/planner/exceptions.py - Error hierarchy                  
     4. src/core/orchestrator/planner/prompts.py - LLM prompt engineering              
     5. src/core/orchestrator/planner/interfaces.py - Abstract base classes            
     6. src/core/orchestrator/planner/planner.py - Main orchestrator                   
                                                                                       
     Must test:                                                                        
     1. tests/unit/test_schemas.py                                                     
     2. tests/unit/test_validators.py                                                  
     3. tests/integration/test_planner.py                                              
                                                                                       
     Reference files (existing patterns to match):                                     
     - src/core/orchestrator/planner/plan.py (OpenRouter client pattern)               
     - src/core/orchestrator/solver/solver.py (Function structure)                     
     - src/core/python_interpreter-e2b/main.py (Async sandbox pattern)                 
                                                                                       
     ---                                                                               
     Next Steps After Planner Complete                                                 
                                                                                       
     Once planner.py is implemented and tested:                                        
                                                                                       
     1. Implement Critics (critic.py):                                                 
       - Physics Lawyer (unit consistency, law applicability)                          
       - Dependency Checker (variable dependencies, cycles)                            
       - Pre-Mortem (identify most likely failure point)                               
     2. Implement Gatekeeper (reviewer.py):                                            
       - Boolean judge (approve/reject based on critique severity)                     
       - Max 2 critique-refine iterations                                              
     3. Update orchestrate.py:                                                         
       - Wire planner → critics → solver → cleaner                                     
       - Implement MDAP voting loop                                                    
       - Connect to e2b sandbox                                                        
     4. Build Cleaner/Validator:                                                       
       - Post-execution validation                                                     
       - Unit checking                                                                 
       - Commit to StateObject.knowns                                                  
     5. Benchmarking:                                                                  
       - PhysReason benchmark                                                          
       - Custom test suite (degrees/radians traps)                                     
       - Error rate vs cost analysis                                                   
                                                                                       
     ---                                                                               
     References                                                                        
                                                                                       
     Spec document sections:                                                           
     - Section II: Architecture (components + flow)                                    
     - Section III: Concrete schemas and prompts                                       
     - Section IV: Justification (MAKER, Self-Consistency, CoVe papers)                
     - Section V: Telemetry and ablations                                              
                                                                                       
     Papers cited:                                                                     
     - MAKER (MDAP): arxiv.org/pdf/2511.09030                                          
     - Self-Consistency: arxiv.org/pdf/2203.11171                                      
     - Multi-Agent Critique:                                                           
     emergentmind.com/topics/multi-agent-critique-and-revision                         
                                                                                       
     Existing code patterns:                                                           
     - plan.py:14-37 (OpenRouter client initialization)                                
     - solver.py:14-37 (LLM call pattern)                                              
     - voting_pool.py:1 (k=3 for voting threshold) 