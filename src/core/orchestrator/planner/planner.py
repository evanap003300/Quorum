"""
Main planner orchestration for the Quorum planning system.

This module provides the Planner class, which:
1. Generates atomic plans from natural language problems
2. Integrates with critics for plan validation
3. Manages the critique-refine loop
4. Produces structured StateObject outputs

The Planner is the main entry point for planning. It orchestrates all other components:
- Schemas: Data models with strict validation
- Validators: Atomicity and structure checking
- Prompts: LLM prompt engineering
- Interfaces: Abstract base classes for critics/gatekeeper
- Exceptions: Custom error hierarchy
"""

import os
import json
import logging
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from .schemas import StateObject, Plan, CritiqueUnit
from .interfaces import BaseCritic, BaseGatekeeper
from .validators import validate_plan_structure, validate_all_steps_atomic
from .prompts import (
    get_planning_system_prompt,
    get_planning_user_prompt,
    get_refinement_system_prompt,
    get_refinement_user_prompt,
)
from .exceptions import (
    PlanningError,
    SchemaValidationError,
    LLMError,
    CritiqueError,
    RefinementError,
)

# Configure logging
logger = logging.getLogger(__name__)


class Planner:
    """
    Self-contained planner that generates atomic, validated plans from natural language.

    The Planner orchestrates the full planning pipeline:
    1. Parse problem → StateObject
    2. Generate initial plan via LLM
    3. Validate schema and atomicity
    4. Run critique-refine loop (max 2 iterations)
    5. Return validated StateObject with approved Plan

    Example:
        planner = Planner()
        state = planner.create_plan(
            "A ball is thrown at 20 m/s. How long until it lands?",
            domain="physics"
        )
        print(f"Generated {len(state.plan.steps)} steps")

    Configuration:
        model: LLM model to use (default: "openai/gpt-4o")
        temperature: LLM temperature (default: 0.1 for determinism)
        max_retries: Max retries on schema failure (default: 2)
        critics: List of BaseCritic instances (optional)
        gatekeeper: BaseGatekeeper instance (optional)
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        temperature: float = 0.1,
        max_retries: int = 2,
        critics: Optional[List[BaseCritic]] = None,
        gatekeeper: Optional[BaseGatekeeper] = None,
    ):
        """
        Initialize the Planner.

        Args:
            model: OpenRouter model ID (e.g., "openai/gpt-4o")
            temperature: LLM temperature (0.0 = deterministic, 1.0 = random)
            max_retries: Max retries on schema validation failure (0-5 recommended)
            critics: List of critic instances to run (optional)
            gatekeeper: Gatekeeper instance for plan approval (optional)

        Raises:
            ValueError: If invalid parameters provided
            EnvironmentError: If OPEN_ROUTER_KEY not found
        """
        # Validate parameters
        if not 0 <= temperature <= 1:
            raise ValueError(f"temperature must be in [0, 1], got {temperature}")
        if max_retries < 0 or max_retries > 5:
            raise ValueError(f"max_retries should be in [0, 5], got {max_retries}")

        # Load environment
        load_dotenv()
        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPEN_ROUTER_KEY not found in environment. "
                "Set it in .env file or as environment variable."
            )

        # Initialize OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Store configuration
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.critics = critics or []
        self.gatekeeper = gatekeeper

        # Logging
        logger.info(
            f"Initialized Planner: model={model}, temp={temperature}, "
            f"retries={max_retries}, critics={len(self.critics)}"
        )

    def create_plan(self, problem_text: str, domain: str = "physics") -> StateObject:
        """
        Main entry point: Generate a validated plan from a problem description.

        Pipeline:
        1. Create initial StateObject
        2. Generate plan via LLM (with retries)
        3. Run critique-refine loop if critics present
        4. Return approved StateObject with Plan

        Args:
            problem_text: Natural language problem description
            domain: Problem domain ('physics', 'mathematics', 'chemistry', etc.)

        Returns:
            StateObject with generated and validated Plan

        Raises:
            PlanningError: If planning fails after max retries
            ValueError: If problem_text is empty
        """
        # Validate input
        if not problem_text or not problem_text.strip():
            raise ValueError("problem_text cannot be empty")

        # Initialize state object
        state = StateObject(
            problem_text=problem_text,
            domain=domain,
            goal="",  # Will be filled by LLM
        )
        state.log_event("planning_started", f"Problem: {problem_text[:100]}...")
        logger.debug(f"Created StateObject {state.state_id}")

        try:
            # Phase 1: Generate initial plan
            plan = self._generate_plan_with_retries(state)
            state.plan = plan
            state.log_event("plan_generated", f"Generated plan v{plan.plan_version} with {len(plan.steps)} steps")
            logger.info(f"Successfully generated plan with {len(plan.steps)} steps")

            # Phase 2: Run critique-refine loop if critics available
            if self.critics and self.gatekeeper:
                logger.debug(f"Running critique-refine loop with {len(self.critics)} critics")
                state = self._critique_refine_loop(state)
            else:
                logger.debug("No critics/gatekeeper configured; skipping critique loop")

            return state

        except Exception as e:
            state.log_event("planning_failed", str(e))
            logger.error(f"Planning failed: {e}")
            raise

    def _generate_plan_with_retries(self, state: StateObject) -> Plan:
        """
        Generate a plan with schema validation and retry logic.

        Attempts to generate a valid Plan JSON up to (max_retries + 1) times.
        On schema failure, includes error message in retry prompt to guide LLM.

        Args:
            state: The StateObject with problem context

        Returns:
            Validated Plan object

        Raises:
            PlanningError: If all retries exhausted
            SchemaValidationError: If LLM output invalid (after retries)
            LLMError: If LLM API call fails
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Plan generation attempt {attempt + 1}/{self.max_retries + 1}")

                # Build LLM messages
                messages = [
                    {"role": "system", "content": get_planning_system_prompt()},
                    {"role": "user", "content": get_planning_user_prompt(state)}
                ]

                # If retrying, add previous error info
                if attempt > 0 and last_error:
                    messages.append({
                        "role": "assistant",
                        "content": f"(Previous attempt failed with error: {str(last_error)[:200]}. Please try again with valid JSON.)"
                    })

                # Call LLM
                logger.debug(f"Calling LLM: {self.model} with temperature={self.temperature}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1000,
                    response_format={"type": "json_object"}  # Force JSON output
                )

                # Extract and parse JSON
                raw_output = response.choices[0].message.content
                logger.debug(f"LLM response (first 200 chars): {raw_output[:200]}")

                try:
                    plan_data = json.loads(raw_output)
                except json.JSONDecodeError as e:
                    last_error = SchemaValidationError(
                        f"LLM output is not valid JSON: {str(e)}",
                        raw_output=raw_output
                    )
                    logger.warning(f"Attempt {attempt + 1}: JSON decode error: {last_error.message}")
                    if attempt < self.max_retries:
                        continue
                    else:
                        raise PlanningError(
                            f"Failed to generate valid JSON after {self.max_retries + 1} attempts"
                        ) from last_error

                # Validate with Pydantic
                try:
                    plan = Plan(**plan_data)
                except ValueError as e:
                    last_error = SchemaValidationError(
                        f"Plan data does not match schema: {str(e)}",
                        raw_output=raw_output
                    )
                    logger.warning(f"Attempt {attempt + 1}: Schema validation error: {last_error.message}")
                    if attempt < self.max_retries:
                        continue
                    else:
                        raise PlanningError(
                            f"Failed to generate valid plan after {self.max_retries + 1} attempts"
                        ) from last_error

                # Additional structural validation
                try:
                    validate_plan_structure(plan)
                    validate_all_steps_atomic(plan)
                except (ValueError, Exception) as e:
                    last_error = SchemaValidationError(
                        f"Plan structure validation failed: {str(e)}",
                        raw_output=raw_output
                    )
                    logger.warning(f"Attempt {attempt + 1}: Structure validation error: {last_error.message}")
                    if attempt < self.max_retries:
                        continue
                    else:
                        raise PlanningError(
                            f"Plan failed structural validation after {self.max_retries + 1} attempts"
                        ) from last_error

                # Success!
                logger.info(f"Successfully generated valid plan on attempt {attempt + 1}")
                return plan

            except LLMError as e:
                last_error = e
                logger.error(f"Attempt {attempt + 1}: LLM error: {e}")
                if attempt < self.max_retries:
                    continue
                else:
                    raise

            except Exception as e:
                last_error = LLMError(f"Unexpected error in plan generation: {str(e)}")
                logger.error(f"Attempt {attempt + 1}: Unexpected error: {e}")
                if attempt < self.max_retries:
                    continue
                else:
                    raise PlanningError(
                        f"Plan generation failed unexpectedly: {str(e)}"
                    ) from last_error

        # All retries exhausted
        raise PlanningError(
            f"Failed to generate valid plan after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def _critique_refine_loop(self, state: StateObject, max_iterations: int = 2) -> StateObject:
        """
        Run the critique-refine loop with gatekeeper approval.

        Process:
        1. Collect critiques from all critics
        2. Gatekeeper decides if refinement needed
        3. If refinement needed, refine plan and loop (max 2 iterations)
        4. Otherwise approve and return

        Args:
            state: The StateObject with plan to critique
            max_iterations: Maximum refinement iterations (default: 2)

        Returns:
            StateObject with refined/approved plan

        Raises:
            CritiqueError: If critique process fails
        """
        logger.debug(f"Starting critique-refine loop (max {max_iterations} iterations)")

        for iteration in range(max_iterations):
            logger.debug(f"Critique-refine iteration {iteration + 1}/{max_iterations}")

            # Collect critiques from all critics
            all_critiques = []
            try:
                for critic in self.critics:
                    logger.debug(f"Running critic: {critic.critic_id}")
                    critiques = critic.critique(state.plan)
                    all_critiques.extend(critiques)
                    logger.debug(f"  → {len(critiques)} critique(s)")
            except Exception as e:
                raise CritiqueError(
                    f"Critique collection failed: {str(e)}",
                    critic_id=getattr(critic, 'critic_id', 'unknown')
                ) from e

            state.log_event("critiques_collected", f"Collected {len(all_critiques)} critique(s)")

            if not all_critiques:
                logger.info("No critiques found; plan approved")
                state.log_event("plan_approved", "No issues from critics")
                break

            # Gatekeeper decides if refinement needed
            try:
                should_refine, blocking_issues = self.gatekeeper.should_refine(all_critiques)
                logger.debug(f"Gatekeeper decision: should_refine={should_refine}, blocking={len(blocking_issues)}")
            except Exception as e:
                raise CritiqueError(f"Gatekeeper evaluation failed: {str(e)}") from e

            if not should_refine:
                logger.info(f"Gatekeeper approved plan despite {len(all_critiques)} critique(s)")
                state.log_event("plan_approved", f"Approved with {len(all_critiques)} non-blocking issues")
                break

            # Refine plan based on critiques
            logger.debug(f"Refining plan due to {len(blocking_issues)} blocking issue(s)")
            state.log_event("plan_refinement", f"Refinement {iteration + 1}: {len(blocking_issues)} blocking issues")

            try:
                refined_plan = self._refine_plan(state, all_critiques)
                state.plan = refined_plan
            except Exception as e:
                raise RefinementError(
                    f"Plan refinement failed: {str(e)}",
                    iteration=iteration + 1,
                    attempts=iteration + 1
                ) from e

        logger.debug("Critique-refine loop complete")
        return state

    def _refine_plan(self, state: StateObject, critiques: List[CritiqueUnit]) -> Plan:
        """
        Refine a plan based on critic feedback.

        Sends the plan and critiques to the LLM with a refinement prompt,
        asking it to fix the issues while maintaining atomicity.

        Args:
            state: The StateObject with current plan
            critiques: List of CritiqueUnit feedback items

        Returns:
            Refined Plan object

        Raises:
            SchemaValidationError: If refined plan doesn't validate
            LLMError: If LLM call fails
        """
        logger.debug(f"Refining plan based on {len(critiques)} critique(s)")

        # Build refinement messages
        messages = [
            {"role": "system", "content": get_refinement_system_prompt()},
            {"role": "user", "content": get_planning_user_prompt(state)},
            {"role": "assistant", "content": json.dumps(state.plan.model_dump(), default=str)},
            {
                "role": "user",
                "content": get_refinement_user_prompt(
                    json.dumps(state.plan.model_dump(), indent=2, default=str),
                    json.dumps([c.model_dump() for c in critiques], indent=2, default=str)
                )
            }
        ]

        # Call LLM with lower temperature for focused refinement
        logger.debug("Calling LLM for plan refinement (temperature=0.0)")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,  # Deterministic refinement
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        # Parse and validate refined plan
        raw_output = response.choices[0].message.content

        try:
            plan_data = json.loads(raw_output)
            refined_plan = Plan(**plan_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise SchemaValidationError(
                f"Refined plan validation failed: {str(e)}",
                raw_output=raw_output
            )

        # Increment version
        refined_plan.plan_version = state.plan.plan_version + 1

        # Validate structure
        try:
            validate_plan_structure(refined_plan)
            validate_all_steps_atomic(refined_plan)
        except Exception as e:
            raise SchemaValidationError(
                f"Refined plan failed structural validation: {str(e)}",
                raw_output=raw_output
            )

        logger.info(f"Successfully refined plan to v{refined_plan.plan_version}")
        return refined_plan


# Export Planner class
__all__ = ["Planner"]
