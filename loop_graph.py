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

from static_objects import game,game_string_format
from game_functions import add_or_change_character,add_or_change_item_to_character_inventory,delete_item_from_character_inventory,define_story,roll_dice

import os

class State(TypedDict):
        messages: Annotated[list, add_messages]
        tool_controller_messages: Annotated[list, add_messages]

tools = [
    add_or_change_character,
    add_or_change_item_to_character_inventory,
    delete_item_from_character_inventory,
    define_story,
    roll_dice
]

# llm = init_chat_model("google_genai:gemini-2.0-flash")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3-32b",  # veya Groq'un desteklediği başka bir model
    reasoning_effort="none",
)

llm_with_tools = llm.bind_tools(tools)

def looper(state: State):
    print("chatbot stage")
    
    message = llm.invoke(state["messages"])

    return {"messages": [message]}

def tool_controller(state: State):
    print("tool_controller stage")
    # Only take the last 4 messages, or fewer if there aren't enough
    recent_messages = state["messages"][-4:]

    chat_prompt = ChatPromptTemplate.from_messages([
            tool_system_message,
            *recent_messages,
        ])
    
    message = llm_with_tools.invoke(chat_prompt.format_messages())

    return {"tool_controller_messages": [message]}

last_x_rounds = 10



system_message = SystemMessage("You are a FRPG Game Master. " \
f"You will be provided with the game scenario, the game history brief, the rules, the last {last_x_rounds} rounds "\
f"(or the current history if there are less rounds than {last_x_rounds}, a rag history related to the prompt and characters' current inventories etc. " \
"Process the user's input, continue the story and play the NPC's rounds. " \
"If the user asks anything outside the game world (e.g., about real life, modern world facts, technology),"\
"you must stay in character and reply as if you don't understand or bring the conversation back to the fantasy setting." \

"""Please output the game turns in the following structured format:

- Use ALL CAPS for round titles.
- Use bullet points for dialogues and events.
- Always include a final summary section, prefixed with 'SUMMARY:' in bold.
- Use consistent indentation and markdown styling if needed.""")


tool_system_message = SystemMessage("You are an assistant to a FRPG Game Master. " \
f"""You will be provided a couple of rounds. The rounds represent history except for the last round. You have to understand which is the last round first.
Later, you will decide if any of the tools you are provided are necessary or not, in the last round. Later, if you have found one or more of the tools are necessary, please use them correctly.
If one or more of the characters spent or gained money, use reduce_money() or add_money() respectively.
If one or more of the characters gained or lost an item, use add_or_change_item_to_character_inventory() or delete_item_from_character_inventory() respectively.
If a dice roll needs to happen, roll the dice please.
""")


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

        graph_builder.add_node("looper2",looper)

        graph_builder.add_node("tool_controller",tool_controller)

        graph_builder.add_node("tools",tool_stage)

        graph_builder.add_edge(START, "prepare_prompts")
        
        graph_builder.add_edge("prepare_prompts", "looper")

        graph_builder.add_edge("looper", "tool_controller")

        graph_builder.add_conditional_edges(
            "tool_controller",
            tools_condition,
        )

        graph_builder.add_edge("tools","looper2")


        graph = graph_builder.compile(checkpointer=MemorySaver())


        self.graph = graph

# png_data = LoopGraph().graph.get_graph().draw_mermaid_png()

# # PNG'yi dosyaya yaz:
# with open("loop_graph_v4.png", "wb") as f:
#     f.write(png_data)