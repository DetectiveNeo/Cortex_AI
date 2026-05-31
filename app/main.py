import sys
from app.graph.workflow import app_graph

def main():
    """
    Main runner for the Cortex LangGraph workflow.
    Invokes the StateGraph and displays state transitions and the final compiled report.
    """
    print("==================================================")
    print("           STARTING CORTEX WORKFLOW              ")
    print("==================================================")
    
    # Prompt the user directly for their query
    try:
        user_query = input("\nEnter your query/topic to research (or press Enter for default): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nOperation cancelled. Exiting.")
        sys.exit(0)
        
    # Default query fallback if empty input is provided
    if not user_query:
        user_query = "Leveraging LangGraph for Multi-Agent Workflows"
        print(f"Using default query: '{user_query}'")
        
    initial_state = {
        "user_query": user_query
    }
    
    print("\n--------------------------------------------------")
    print(f"Initializing Workflow for: '{user_query}'")
    print("--------------------------------------------------")
    
    # Execute the LangGraph workflow
    final_state = app_graph.invoke(initial_state)
    
    print("\n==================================================")
    print("         WORKFLOW EXECUTION COMPLETED             ")
    print("==================================================")
    
    # Print only the final report, keeping output clean and focused
    print("\n--- FINAL GENERATED REPORT ---")
    report = final_state.get("report")
    if report:
        print(report)
    else:
        print("Error: No report generated in final state.")
        
    print("\n==================================================\n")

if __name__ == "__main__":
    main()
