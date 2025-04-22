from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from tavily import tavily

model = ChatOpenAI(model="gpt-3.5-turbo")

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""


RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


from langchain_core.pydantic_v1 import BaseModel

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revision: int
    

class Queries(BaseModel):
    queries: List[str]

from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
from tavily import TavilyClient
import os
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def plan_node(state:AgentState):
    messages = [SystemMessage(content= PLAN_PROMPT),
                HumanMessage(content= state['task'])
               ]
    response = model.invoke(messages)
    print(f"plan_node: {response}")
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(queries).invoke([
        SystemMessage(content = RESEARCH_PLAN_PROMPT),
        HumanMessage(content = state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results = 2)
        print(f"research_plan_node: {response}")
        for r in response['result']:
            content.append(r['content'])

def generate_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_response = HumanMessage(content = f"{state['task']} \nHere is my plan \n\n{state['plan']}")
    response = model.invoke([SystemMessage(content = WRITER_PROMPT.format(content=content)),
                           user_response])
    print(f"generate_node:{response}")
    return {"draft": response.content, "revision_number": state.get("revision_number")}

def reflection_node(state: AgentState):
    message = [SystemMessage(content = REFLECTION_PROMPT),
               HumanMessage(content = state['draft'])]
    response = model.invoke(message)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(queries).invoke([
        SystemMessage(content = RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content = state['critique'])
    ])
    content = state['content'] or []
    for res in queries.queries:
        response = tavily.search(query = res, max_results = 2)
        for r in response:
            content.append(r['content'])
    return {"content": content}
    

def should_continue(state):
    if state['revision_number'] > state['max_revisions']:
        return END
    return "reflext"
    

builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generate_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)

builder.add_conditional_edges(
    "generate",
    should_continue,
    {"reflect": "reflect", False: END}
    
)

builder.set_entry_point("planner")
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile(checkpointer = memory)

from IPython.display import Image
Image(graph.get_graph().draw_png())

thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({"task": "what is the difference between langchain and langgraph",
                      "revision_number": 1,
                      "max_revision": 2}, thread):
    print(s)
