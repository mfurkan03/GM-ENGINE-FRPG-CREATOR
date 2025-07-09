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
from langchain_core.messages import RemoveMessage

from pydantic import BaseModel,Field

from static_objects import game,game_string_format,BasicToolNode
from game_functions import add_or_change_character,add_or_change_item_to_character_inventory,delete_item_from_character_inventory,define_story,roll_dice,add_money,reduce_money

import os

class State(TypedDict):
        messages: Annotated[list, add_messages]
        tool_controller_messages: Annotated[list, add_messages]

class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    tool_called : bool = Field(
        description="If there are any tools called",
    )

    reason: str = Field(
        description="Reasons and explanation of the tool calls and or calling them.",
    )

    summary: str = Field(
        description="Summary of the action that required tools if there are any, within a sentence.",
    )

tools = [
    add_or_change_character,
    add_or_change_item_to_character_inventory,
    delete_item_from_character_inventory,
    add_money,
    reduce_money,
    ResponseFormatter
]


llm = init_chat_model("google_genai:gemini-2.5-flash")

# llm = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model="llama3-70b-8192",  # veya Groq'un desteklediği başka bir model
# )

last_x_rounds = 10

system_message = SystemMessage("You are a FRPG Game Master. " \
f"You will be provided with the game scenario, the game history brief, the rules, the last {last_x_rounds} rounds "\
f"(or the current history if there are less rounds than {last_x_rounds}, a rag history related to the prompt and characters' current inventories etc. " \
"Process the user's input, continue the story and ONLY play the NPC's rounds. Do not ask any of the NPCs moves, do them all by yourself. " \
"If the user asks anything outside the game world (e.g., about real life, modern world facts, technology),"\
"you must stay in character and reply as if you don't understand or bring the conversation back to the fantasy setting." \
)

tool_system_message = SystemMessage("You are an assistant to a FRPG Game Master. " \
f"""You will be provided a couple of rounds. 
You will decide if any of the tools you are provided are necessary or not. Lastly, if you have found that one or more of the tools are necessary, please use them correctly.
If one or more of the characters spent or gained money, use reduce_money() or add_money() respectively.
If one or more of the characters gained or lost an item, or made transaction, use add_or_change_item_to_character_inventory() or delete_item_from_character_inventory() respectively.
""")


llm_with_tools = llm.bind_tools(tools)


def looper(state: State):
    print("chatbot stage")
    
    message = llm.invoke(state["messages"])

    return {"messages": [message]}

def tool_controller(state: State):
    print("tool_controller stage")
    # Only take the last 4 messages, or fewer if there aren't enough
    recent_messages = state["messages"][-4:]

    recent_messages.append("""First, find the last round or last action made. On that last round, decide if any of the tools you are provided are necessary or not in this situation. Explain which of them are necessary with a short explanation and the reasons for that. If you have found that one or more of the tools are necessary, use them correctly. If it wasn't last round, than don't use the tools. Look only at last round.
If one or more of the characters spent or gained money, use reduce_money() or add_money() respectively.
If one or more of the characters gained or lost an item, or made transaction, use add_or_change_item_to_character_inventory() or delete_item_from_character_inventory() respectively. I want you to not use tools twice for the same round, therefore, only evaluate the last round.
Use add or change character only if a character enters our party permanently.
If you use a price is not given for a transaction, try to deduce it first.""")

    chat_prompt = ChatPromptTemplate.from_messages([
            *recent_messages,
        ])

    message = llm_with_tools.invoke(chat_prompt.format_messages())

    return {"messages": [message]}
    
def filter_out_rule_messages(state:State) -> Annotated[list, "Filtered messages with game rules removed"]:
    """
    Filters out any HumanMessage that starts with the current game rules text.

    Args:
        messages (list): List of message objects to filter.

    Returns:
        list: Filtered list of messages, excluding HumanMessages that repeat the game rules.
    """

    messages = state["messages"]

    _dict = {"messages": [RemoveMessage(id=msg.id) for msg in messages if msg.content.startswith("The rules are ")]}

    return _dict 

def prepare_prompts_node(state):
    task = state["messages"][-1]

    human_msg_2 = HumanMessage(f"The rules are {game.rules}. The current characters and their current inventories stats and other details are : {game.characters}")

    chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            human_msg_2,
            task,
        ])

    

    formatted_messages = chat_prompt.format_messages()

    # State'e ekle
    return {"messages": formatted_messages}

def tool_stage(state:State):
    print("tool stage")
    tool_node = ToolNode(tools=tools)
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
    return END

class LoopGraph:
     
    def __init__(self):
        
        self.config = {"configurable": {"thread_id": "2"}}

        graph_builder = StateGraph(State)
        
        graph_builder.add_node("filter",filter_out_rule_messages)

        graph_builder.add_node("prepare_prompts",prepare_prompts_node)

        graph_builder.add_node("looper",looper)

        graph_builder.add_node("tool_controller",tool_controller)

        graph_builder.add_node("tools",tool_stage)

        graph_builder.add_edge(START, "filter")
        
        graph_builder.add_edge( "filter","prepare_prompts")

        graph_builder.add_edge("prepare_prompts", "looper")

        graph_builder.add_edge("looper","tool_controller")

        graph_builder.add_conditional_edges(
            "tool_controller",route_tools
            ,{"tools":"tools",END:END}
        )

        graph_builder.add_edge("tools",END)

        graph = graph_builder.compile(checkpointer=MemorySaver())

        self.graph = graph

# png_data = LoopGraph().graph.get_graph().draw_mermaid_png()

# # PNG'yi dosyaya yaz:
# with open("graph_png/loop_graph_v7.png", "wb") as f:
#     f.write(png_data)