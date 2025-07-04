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
from game_functions import add_character,add_item_to_character_inventory,delete_item_from_character_inventory

import os


class State(TypedDict):
        messages: Annotated[list, add_messages]


tools = [
    add_character,
    add_item_to_character_inventory,
    delete_item_from_character_inventory
]

llm = init_chat_model("google_genai:gemini-2.0-flash")

llm_with_tools = llm.bind_tools(tools)

def looper(state: State):
    print("chatbot stage")
    
    message = llm_with_tools.invoke(state["messages"])

    return {"messages": [message]}

last_x_rounds = 10

system_message = SystemMessage("You are a FRPG Game Master. " \
"You will be provided with the game scenario, the game history brief, the rules, the last {last_x_rounds} rounds if there are that much and, a rag history related to the prompt. " \
"Process the user's input, continue the story and play the NPC's rounds.")

def prepare_prompts_node(state):
    task = state["messages"][-1]

    chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            task,
        ])

    formatted_messages = chat_prompt.format_messages()

    # State'e ekle
    return {"messages": formatted_messages}

def tool_stage(state:State):
    tool_node = ToolNode(tools=tools)
    print("tool stage")
    return tool_node


class LoopGraph:
     
    def __init__(self):
        
        self.config = {"configurable": {"thread_id": "2"}}

        graph_builder = StateGraph(State)
        
        graph_builder.add_node("prepare_prompts",prepare_prompts_node)

        graph_builder.add_node("looper",looper)

        graph_builder.add_node("tools",tool_stage)

        graph_builder.add_edge(START, "prepare_prompts")
        
        graph_builder.add_edge("prepare_prompts", "chatbot")

        graph_builder.add_conditional_edges(
            "looper",
            tools_condition,
        )
        
        graph_builder.add_edge("chatbot",END)

        graph = graph_builder.compile(checkpointer=MemorySaver())


        self.graph = graph

