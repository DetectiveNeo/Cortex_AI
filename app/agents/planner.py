from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from app.models.state import CortexState

# 1. Define the Pydantic schema for the structured LLM output
class ResearchPlan(BaseModel):
    """
    A structured plan containing a list of specific research tasks.
    """
    tasks: list[str] = Field(
        description="A list of 3 to 5 clear, sequential, and actionable research tasks designed to answer the user query."
    )

def planner(state: CortexState) -> dict:
    """
    Planner agent node. Analyzes the user query and generates a structured plan
    of 3 to 5 research tasks using ChatOpenAI's structured output.
    """
    print("\n--- [PLANNER AGENT] Starting execution ---")
    query = state.user_query
    print(f"Received User Query: '{query}'")
    
    tasks = []
    
    try:
        # Initialize LLM with standard parameters
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # Bind the Pydantic schema for structured JSON output
        structured_llm = llm.with_structured_output(ResearchPlan)
        
        system_prompt = (
            "You are an expert research planner. Given a user query, "
            "generate a structured list of 3 to 5 clear, sequential, "
            "and actionable research tasks to thoroughly investigate the query."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        print("Invoking ChatOpenAI for structured plan generation...")
        result = structured_llm.invoke(messages)
        
        # Extract tasks from Pydantic structured response
        if result and hasattr(result, "tasks") and result.tasks:
            tasks = result.tasks
            print(f"Successfully generated {len(tasks)} research tasks via structured output.")
        else:
            raise ValueError("Empty or invalid structured output received.")
            
    except Exception as e:
        print(f"Warning: OpenAI LLM structured generation failed or API key missing ({e}).")
        print("Falling back to robust static structured tasks...")
        # Graceful fallback logic to satisfy the 3-5 research tasks requirement
        tasks = [
            f"Define core concepts, terminology, and definitions related to '{query}'",
            f"Analyze recent industry trends, top publications, and case studies about '{query}'",
            f"Identify standard methodologies, practices, and tooling currently used for '{query}'",
            f"Compile a list of primary challenges, architectural bottlenecks, and mitigation strategies for '{query}'"
        ]
        
    print("Generated Plan:")
    for idx, task in enumerate(tasks, 1):
        print(f"  {idx}. {task}")
        
    # Store the list of tasks directly in state.plan
    return {"plan": tasks}
