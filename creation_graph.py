from typing import Annotated, Literal

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_ollama.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain.prompts import ChatPromptTemplate

from pydantic import BaseModel,Field

from static_objects import game,first_prompt
from game_functions import add_or_change_character,add_or_change_item_to_character_inventory,define_rules,define_story

import os

class State(TypedDict):
        messages: Annotated[list, add_messages]
        format_comply_or_not:str
        feedback:str
        current_task:str
        current_schema_no:int

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3-32b",  # veya Groq'un desteklediği başka bir model
)

#llm = ChatOllama(model = "llama3.2:3b")


tools = [
    add_or_change_character,
    add_or_change_item_to_character_inventory,
    define_rules,
    define_story
]

# llm = init_chat_model("google_genai:gemini-2.0-flash")

llm_with_tools = llm.bind_tools(tools)

class Feedback(BaseModel):

    format_comply_or_not: Literal["comply", "not comply"] = Field(
        description="Decide if the output complies the format or not.",
    )

    feedback: str = Field(
        description="You are an evaluator for an FRPG game. The user will send you a prompt regarding the requested format and an output for that format, and you will check if the produced output fits the format. Were the requests met? Are there blank fields that should not be blank? Don't be too harsh!",
    )

evaluator = llm.with_structured_output(Feedback)

def chatbot(state: State):
    print("chatbot stage")
    if state.get("feedback"):
        state["messages"].append("The format is wrong because: "+state.get("feedback")+", please correct it.")
        message = llm_with_tools.invoke(state["messages"])
    
    else:  
        
        message = llm_with_tools.invoke(state["messages"])
    
    return {"messages": [message]}

def evaluator_stage(state:State):
    print("evaluator stage")
    schema = [game.story,game.rules,game.characters,game.characters]
    schema = schema[state["current_schema_no"]]
    human_message = "The requested task is this:" + state["current_task"]+"The output generated for this task is this:"+str(schema)+" Are there any mistakes here? If so, please specify the source of the mistake. Else, Answer with yes. Note, the tools that were available in the task are not available for you, don't take tool usage into consideration."

    response = evaluator.invoke([SystemMessage("You are an evaluator for an FRPG game. The user will send you a prompt regarding the requested format and an output for that format, and you will check if the produced output fits the format. Are there blank fields that should not be blank? Don't be too harsh the format doesn't have to exactly comply. If there isn't any blank field or structural mistake, then no problem!"),HumanMessage(human_message)])
    
    # print(response.feedback,game.characters,game.rules)
    print(response)
    return {"format_comply_or_not":response.format_comply_or_not, "feedback":response.feedback}

def route_feedback(state:State):
    
    return state["format_comply_or_not"]

def tool_stage(state:State):
    tool_node = ToolNode(tools=tools)
    print("tool stage")
    return tool_node

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the evaluator.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "evaluator"

system_message = SystemMessage("You are a professional FRPG game designer.")

def prepare_prompts_node(state):
    task = state["messages"][-1]

    chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            task,
        ])

    formatted_messages = chat_prompt.format_messages()

    # State'e ekle
    return {"messages": formatted_messages}

class FullGraph:
     
    def __init__(self):
        
        self.config = {"configurable": {"thread_id": "4"}, "recursion_limit":25}

        graph_builder = StateGraph(State)

        graph_builder.add_node("prepare_prompts", prepare_prompts_node)

        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("tools", tool_stage)
        graph_builder.add_node("evaluator", evaluator_stage)

        graph_builder.add_edge(START, "prepare_prompts")
        graph_builder.add_edge("prepare_prompts", "chatbot")

        graph_builder.add_conditional_edges(
            "chatbot",
            route_tools, {
                "tools":"tools",
            
                "evaluator": "evaluator"
                },
        )
        
        
        graph_builder.add_conditional_edges(
            "evaluator",
            route_feedback,
            {
                "comply":END,
            
                "not comply": "chatbot"
                },
        )

        graph_builder.add_edge("tools", "evaluator")

        graph = graph_builder.compile()

        self.graph = graph


# png_data = FullGraph().graph.get_graph().draw_mermaid_png()

# # PNG'yi dosyaya yaz:
# with open("full_graph.png", "wb") as f:
#     f.write(png_data)