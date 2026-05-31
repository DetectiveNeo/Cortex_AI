from pydantic import BaseModel, Field

class CortexState(BaseModel):
    """
    State definition for the Cortex sequential workflow.
    Tracks user query and progressive results of each pipeline stage.
    """
    user_query: str = Field(default="", description="The query input by the user to plan, research, and report on.")
    plan: list[str] = Field(default_factory=list, description="The execution plan (list of research tasks) designed by the planner agent.")
    research: str = Field(default="", description="The compiled research collected by the researcher agent.")
    summary: str = Field(default="", description="The synthesized summary of the research collected by the summarizer agent.")
    report: str = Field(default="", description="The final comprehensive report compiled by the reporter agent.")
