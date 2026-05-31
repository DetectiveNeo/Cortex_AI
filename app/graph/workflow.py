from langgraph.graph import StateGraph, START, END
from app.models.state import CortexState
from app.agents.planner import planner
from app.agents.researcher import researcher
from app.agents.summarizer import summarizer
from app.agents.reporter import reporter

# 1. Initialize StateGraph using the Pydantic CortexState schema
workflow = StateGraph(CortexState)

# 2. Add nodes representing our placeholder agent processors
workflow.add_node("planner", planner)
workflow.add_node("researcher", researcher)
workflow.add_node("summarizer", summarizer)
workflow.add_node("reporter", reporter)

# 3. Add sequence transitions (START -> planner -> researcher -> summarizer -> reporter -> END)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "summarizer")
workflow.add_edge("summarizer", "reporter")
workflow.add_edge("reporter", END)

# 4. Compile the state workflow graph to generate an executable runnable
app_graph = workflow.compile()
