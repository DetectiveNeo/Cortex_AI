from langchain_openai import ChatOpenAI
from app.models.state import CortexState

def reporter(state: CortexState) -> dict:
    """
    Reporter agent node. Drafts a final report using the summarized insights.
    """
    print("\n--- [REPORTER AGENT] Starting execution ---")
    summary = state.summary
    query = state.user_query
    
    if not summary:
        print("Warning: No summary found in state. Storing empty report.")
        return {"report": "# Empty Report\nNo research summary available to compile a report."}
        
    print("Compiling final comprehensive Markdown report...")
    
    report = ""
    
    try:
        # Initialize LLM with standard parameters
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        system_prompt = (
            "You are an expert technical writer. Your task is to take a research summary "
            "and compile it into a comprehensive, beautifully structured Markdown report. "
            "The report should look highly professional, with clear headings, bullets, "
            "and clean Markdown formatting."
        )
        
        user_prompt = (
            f"Overall Objective/User Query: '{query}'\n\n"
            f"Synthesized Research Summary:\n"
            f"==================================\n"
            f"{summary}\n"
            f"==================================\n\n"
            f"Please generate a complete, formal Markdown report. Use appropriate markdown headers, "
            f"such as an Executive Summary, Key Insights, and Strategic Recommendations."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print("Calling OpenAI to generate Markdown report...")
        response = llm.invoke(messages)
        report = response.content.strip()
        print(f"Markdown report generated successfully ({len(report)} chars).")
        
    except Exception as e:
        print(f"Warning: OpenAI LLM call failed or API credentials missing ({e}).")
        print("Falling back to robust static markdown report compiler...")
        
        # Compile a robust, beautiful Markdown report using state fields
        formatted_plan = "\n".join(f"- {task}" for task in state.plan)
        report = (
        f"==================================================\n"
        f"             CORTEX ANALYSIS REPORT               \n"
        f"==================================================\n"
        f"Subject: {state.user_query.upper()}\n\n"
        f"1. EXECUTIVE SUMMARY\n"
        f"   {summary}\n\n"
        f"2. ROADMAP PLAN\n"
        f"   {formatted_plan}\n\n"
        f"3. COMPILED RESEARCH REFERENCE\n"
        f"   {state.research}\n"
        f"=================================================="
        )
    print("Final Report compiled successfully.")
        
    return {"report": report}
