You have covered the "Big Three" (Execution, Vision, Advanced Math).

To make your system a true "Physics Engine" rather than just a smart calculator, you should add tools that address **Knowledge Retrieval** and **Verification**.

Here is a breakdown of the best general tools to add next.

### 1. The Knowledge Suite (Stop Hallucinations)

Your LLM knows how to *solve* equations, but it is terrible at remembering *facts*. It will often confidently say "The density of osmium is 22.4 g/cm³" (it's 22.59) or hallucinate the moment of inertia for obscure shapes.

#### A. `scipy.constants` Wrapper (The "Fact Checker")

* **What it is:** A simple Python tool that queries the `scipy.constants` library.
* **Why:** It provides **exact**, scientifically accepted values for fundamental constants (Planck's constant, Boltzmann constant, speed of light, proton mass) without relying on the LLM's memory.
* **Agent usage:** "I need the precise mass of a neutron for this calculation. Tool: `get_constant('neutron_mass')`."

#### B. The "Formula RAG" (The "Cheat Sheet")

* **What it is:** A small, curated database (JSON or Vector Store) containing the 500 most common physics formulas (e.g., "Moment of Inertia of a thick spherical shell", "Maxwell's Equations in matter").
* **Why:** LLMs often mix up coefficients (e.g., is it  or ?). A lookup tool allows the agent to "open the textbook" and check the formula before deriving.
* **Agent usage:** "I need the formula for the magnetic field inside a solenoid. Tool: `lookup_formula('solenoid B field')`."

---

### 2. The Validation Suite (Strict Guardrails)

The "Physics Lawyer" uses intuition. These tools use **math** to prove validity.

#### C. Dimensional Analysis Engine (`pint` or `sympy.physics.units`)

* **What it is:** A tool that takes a symbolic formula and the units of its variables, then checks if the equation is dimensionally homogeneous.
* **Why:** This is the single most common way physics students (and LLMs) make mistakes. If the Left Hand Side is [Length] and the Right Hand Side is [Length]/[Time], the derivation is wrong. Period.
* **Agent usage:** "I derived . Let me check the units. Tool: `check_dimensions('x=0.5*a*t', {'x':'m', 'a':'m/s^2', 't':'s'})`." -> **Result:** `False` (Output is [L] vs [L]/[T]).

---

### 3. The Visualization Suite (The "Product" Factor)

Solving the problem is only half the battle. If you want a product humans love, you need to **show** the answer.

#### D. The Plotter (`matplotlib` / `numpy`)

* **What it is:** A tool specifically designed to generate 2D plots of the final solution.
* **Why:**
* **Verification:** A plot reveals physical absurdities instantly (e.g., a projectile going underground).
* **User Experience:** "Here is the trajectory of the ball" is a much better answer than "x(t) = 5t - 4.9t^2".


* **Agent usage:** "I have found the equation of motion. Tool: `plot_function(eq='5*t - 4.9*t**2', range=[0, 10], xlabel='Time', ylabel='Height')`."

---

### 4. The "Search" Suite (The Last Resort)

#### E. Google Search / Serper API

* **What it is:** A tool to search the live web.
* **Why:** Sometimes a problem references a specific real-world object or event that isn't in standard physics data.
* *Example:* "Calculate the force on the spoiler of a **Ferrari F40** moving at 100mph."
* The LLM won't know the F40's drag coefficient or surface area. It needs to Google it.


* **Agent usage:** "I need the drag coefficient of a Ferrari F40. Tool: `web_search('Ferrari F40 drag coefficient')`."

### Recommended Build Order

1. **Dimensional Analysis Tool:** (High Impact). It catches bugs that the Lawyer might miss.
2. **`scipy.constants`:** (Low Effort, High Reliability). Very easy to implement.
3. **The Plotter:** (High "Wow" Factor). Makes your system feel like a sophisticated engine.
