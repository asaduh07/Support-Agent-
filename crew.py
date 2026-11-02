"""CrewAI crew: Retrieval, Lookup and Composer agents (Task 7), structured output (Task 9), least autonomy (Task 15)."""
from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

import llm
from tools import LOOKUP_TOOL_NAME, rag_lookup_tool, ticket_status_tool

LOOKUP_ROLE = "Ticket Lookup Agent"


class TicketInfo(BaseModel):
    record_id: str
    status: str
    category: str
    resolution_time_hours: float
    days_since_created: int = Field(ge=0, le=30)
    escalation_score: float = Field(ge=0, le=1)
    recommend_escalation: bool


class CrewResponse(BaseModel):
    """The schema every crew response must conform to (Task 9)."""
    answer: str = Field(min_length=1)
    grounded: bool  # supported by retrieved KB context or by a ticket record
    sources: list[str]  # KB document ids the answer is drawn from
    ticket: TicketInfo | None = None


def assert_least_autonomy(agents):
    """Application-layer guard (Task 15): only the Lookup Agent may hold the ticket-status tool."""
    for a in agents:
        if a.role != LOOKUP_ROLE and any(t.name == LOOKUP_TOOL_NAME for t in a.tools or []):
            raise PermissionError(f"Least-autonomy violation: '{a.role}' may not use {LOOKUP_TOOL_NAME}; "
                                  f"only '{LOOKUP_ROLE}' may.")


def build_agents(model=None):
    model = model or llm.crew_llm()
    retrieval = Agent(role="Policy Retrieval Agent", goal="Find the Ola support policy that answers the query.",
                      backstory="You search Ola's support-policy knowledge base and never invent policy.",
                      tools=[rag_lookup_tool], llm=model, max_iter=3, verbose=False)
    lookup = Agent(role=LOOKUP_ROLE, goal="Look up the status of any support ticket ID in the query.",
                   backstory="You are the only agent allowed to read support-ticket records.",
                   tools=[ticket_status_tool], llm=model, max_iter=3, verbose=False)
    composer = Agent(role="Response Composer", goal="Write one accurate, grounded reply for the customer.",
                     backstory="You combine policy and ticket facts without adding anything not in them.",
                     tools=[], llm=model, max_iter=2, verbose=False)
    return retrieval, lookup, composer


def build_crew(agents=None):
    retrieval, lookup, composer = agents or build_agents()
    assert_least_autonomy([retrieval, lookup, composer])
    t_rag = Task(description="Use rag_lookup to find the policy relevant to the customer query. "
                             "Return the tool's JSON result unchanged.\nQUERY: {query}",
                 expected_output="The rag_lookup JSON result.", agent=retrieval)
    t_lookup = Task(description="If the query contains a ticket ID (format OLA-0000), call "
                                f"{LOOKUP_TOOL_NAME} with it and return the JSON result unchanged. Otherwise return "
                                '{{"found": false, "record_id": null}}.\nQUERY: {query}',
                    expected_output="The ticket-status JSON result.", agent=lookup)
    t_compose = Task(description="Combine the policy result and the ticket result into one reply to the customer, "
                                 "using only facts present in them.\nQUERY: {query}",
                     expected_output="A CrewResponse JSON object.", agent=composer,
                     context=[t_rag, t_lookup], output_pydantic=CrewResponse)
    return Crew(agents=[retrieval, lookup, composer], tasks=[t_rag, t_lookup, t_compose],
                process=Process.sequential, verbose=False, tracing=False)


def run_crew(query):
    """kickoff() and validate the result against CrewResponse in code."""
    out = build_crew().kickoff(inputs={"query": query})
    return CrewResponse.model_validate(out.pydantic.model_dump() if out.pydantic else out.raw)
