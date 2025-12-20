import base64
import solve from solver.solver
import plan from planner.planner

def generate_solution(question: str, image: base64 = None) -> str:
    context = question + str(image)
    plan(context)
    answer = solve(plan)
    
    return answer