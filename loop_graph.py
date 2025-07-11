from typing import Annotated, Literal

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_ollama.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,ToolMessage,RemoveMessage,AnyMessage
from langchain.prompts import ChatPromptTemplate
from langmem.short_term import SummarizationNode, RunningSummary
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from pydantic import BaseModel,Field

from static_objects import game,game_string_format,BasicToolNode
from game_functions import add_or_change_character,add_or_change_item_to_character_inventory,delete_item_from_character_inventory,define_story,roll_dice,add_money,reduce_money

import queue

import os
import numpy as np

class State(TypedDict):
        messages: Annotated[list, add_messages]
        reason : str
        summary : str
        context: str
        summarized_messages: list[AnyMessage,add_messages]

class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""

    tool_used_other_than_responseformatter: bool = Field(
        description="If any tool other than ResponseFormatter is used, then True."
    )

    reason: str = Field(
        description="Reasons and explanation of the tool calls and or calling them.",
    )

    summary: str = Field(
        description="Summary of the action that required tools if there are any, within a sentence. (Other than this tool)",
    )

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [Document(page_content="LangGraph is a framework for orchestrating LLM apps.",metadata = {"round no":0})]

vector_store = FAISS.from_documents(documents, embedding_model)

tools = [
    add_or_change_character,
    add_or_change_item_to_character_inventory,
    delete_item_from_character_inventory,
    add_money,
    reduce_money,
    ResponseFormatter
]

tooler_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3-32b",  # veya Groq'un desteklediği başka bir model
    reasoning_effort="default",
    reasoning_format="hidden"
)

# llm = init_chat_model("google_genai:gemini-2.0-flash")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",  # veya Groq'un desteklediği başka bir model
)

summarizer_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="gemma2-9b-it",  # veya Groq'un desteklediği başka bir model
)

summarizer_llm = summarizer_llm.bind(max_tokens = 512)

last_x_rounds = 10

system_message = SystemMessage("You are a FRPG Game Master. " \
f"You will be provided with the game scenario, the game history brief, the rules, the last {last_x_rounds} rounds "\
f"(or the current history if there are less rounds than {last_x_rounds}, a rag history related to the prompt and characters' current inventories etc. " \
"Process the user's input, continue the story and ONLY play the NPC's rounds. Do not ask any of the NPCs moves, do them all by yourself. " \
"If the user asks anything outside the game world (e.g., about real life, modern world facts, technology),"\
"you must stay in character and reply as if you don't understand or bring the conversation back to the fantasy setting." \
)

tool_system_message = SystemMessage("You are an assistant to a FRPG Game Master that decides if the tools you are provided are necessary, and uses them if needed." \
f"""
Always use ResponseFormatter tool to format you responses.
""")


llm_with_tools = tooler_llm.bind_tools(tools,parallel_tool_calls=True)

last_actions = []
last_actions.append("No action yet!")

def tool_controller(state: State):
    print("tool_controller stage")
    # Only take the last 4 messages, or fewer if there aren't enough
    recent_messages = state["messages"][-5:]

    recent_messages.append(f"""Instructions for Tool Use and Output Formatting:

Only evaluate the last round or last action.

Ignore earlier rounds.

Do not use any tools if the action is not from the last round.

Avoid duplicate processing:

These actions have already been processed: {last_actions}

Do not repeat any action in this list.

Decide if any tool is necessary:

If yes, explain why and use it correctly.

If no tools are needed, clearly state that and explain why.

Tool usage rules:

Use reduce_money() or add_money() if a character spends or gains money. Explain why.

Use add_or_change_item_to_character_inventory() or delete_item_from_character_inventory() if an item is gained, lost, or traded. Explain the context.

Use add_or_change_character() only if a character permanently joins the party. Explain the reason.

When a transaction is mentioned but no price is explicitly stated, try to infer the price using contextual reasoning. However, do not assume that a transaction has occurred unless it is clearly confirmed.

After any transaction, always update both characters’ inventory and money.

Response formatting:

Always use ResponseFormatter, especially when tools are used.

Follow structured output rules.

Use the character and inventory data provided earlier.

Final instruction:

Explain every step you take in your reasoning, even if you take no action.

""")

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

    _dict = {"messages": [RemoveMessage(id=msg.id) for msg in messages if msg.content.startswith("The main story of my game is") or isinstance(msg,ToolMessage)]}

    return _dict 

def retrieve_rag_result(prompt,last_round):

    # prompt_embedding = embedding_model.embed_query(prompt)
    # last_round_embedding = embedding_model.embed_query(last_round)
    
    # embedding = (np.array(prompt_embedding)+np.array(last_round_embedding))/2

    # results = vector_store.similarity_search_by_vector(embedding, k=3)

    results = vector_store.similarity_search(prompt+last_round,3)

    return results

def prepare_prompts_node(state):
    task = state["messages"][-1]

    context = "Not Provided Yet!"

    if hasattr(state,context):
        context = state["context"]

    prompt = state["messages"][-1].content

    last_round = ""

    if len(state["messages"])>2:
        last_round = state["messages"][-2].content

    result = retrieve_rag_result(prompt,last_round)    

    human_msg_2 = HumanMessage(f"The main story of my game is {game.story}. The rules are {game.rules}. A brief history of the previous events in my game is {context}. The most relevant rounds to this last round according to retrieval augmented generation is :{result} The current characters and their current inventories stats and other details are : {game.characters} I want you to continue the game from this. Evaluate if the user's action makes sense, if it does not, answer accordingly. (Such as trying to swim in the sun or entering a building from a closed window.) If a roll is necessary, do automatically. Here are the last rounds played:")

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

def structure(state:State):
    
    global last_actions

    if messages := state.get("messages", []):
            for i in range(len(messages)-1,0,-1):
                if isinstance(messages[i],AIMessage):
                    message = messages[i]
                    break
            else:
                raise ValueError("No message found in input")
    else:
        raise ValueError("No message found in input")
    
    for tool_call in message.tool_calls:
        print(tool_call["name"])
        if tool_call["name"]=="ResponseFormatter":
            
            args = tool_call["args"]

            pydantic_object = ResponseFormatter.model_validate(args)
            print("bool:",pydantic_object.tool_used_other_than_responseformatter,"reason:",pydantic_object.reason,"summary:",pydantic_object.summary)

            last_actions.insert(0,pydantic_object.summary)
            last_actions = last_actions[:2]
            print("out of structure")
            return {"reason":pydantic_object.reason,"summary":pydantic_object.summary}
    else:
        return {"reason":"No reason!","summary":"No summary!"}

def format_message(message):
    if isinstance(message,HumanMessage):
        role = "Human"
    elif isinstance(message,AIMessage):
        role = "AI"
    else:
        return
    
    return f"{role}: {message.content}"


class LoopGraph:
     
    def __init__(self,max_seen_rounds = 2):
        
        self.config = {"configurable": {"thread_id": "2"}}

        self.round_counter = 0

        self.last_summarized = 0

        self.max_seen_rounds = max_seen_rounds

        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)

        graph_builder = StateGraph(State)
        

        graph_builder.add_node("filter",filter_out_rule_messages)

        graph_builder.add_node("prepare_prompts",prepare_prompts_node)

        graph_builder.add_node("looper",self.looper)

        graph_builder.add_node("tool_controller",tool_controller)

        graph_builder.add_node("tools",tool_stage)

        graph_builder.add_node("structure",structure)

        graph_builder.add_node("prepare_summarize_messages",self.prepare_summarize_messages)

        graph_builder.add_edge(START, "filter")
        
        graph_builder.add_conditional_edges("filter",self.summarize_condition,{"summarize":"prepare_summarize_messages","continue":"prepare_prompts"})

        graph_builder.add_edge("prepare_summarize_messages","prepare_prompts")

        graph_builder.add_edge("prepare_prompts", "looper")

        graph_builder.add_edge("looper","tool_controller")

        graph_builder.add_edge("tool_controller","tools")

        graph_builder.add_edge("tools","structure")

        graph_builder.add_edge("structure",END)

        graph = graph_builder.compile(checkpointer=MemorySaver())

        self.graph = graph


    def prepare_summarize_messages(self,state:State):
        print("summarization stage")

        messages = state["messages"]
        context = "Not Provided Yet!"
        if hasattr(state,"context"):
            context = state["context"]

        messages_to_summarize = []
        
        i = self.last_summarized
        
        while i<self.last_summarized+2:
            if not isinstance(messages[i],ToolMessage):
                messages_to_summarize.append(messages[i])
                i+=1
            
        self.last_summarized = i

        rag_text = ""
        for text in messages_to_summarize:

            formatted = format_message(text)
            
            if formatted:

                rag_text+=formatted

        split_texts = self.splitter.split_text(rag_text)
        
        docs = [Document(page_content=chunk,metadata = {"round no":self.round_counter}) for chunk in split_texts]

        vector_store.add_documents(docs)
        # vector_store.save_local("./data/index_langgraph")

        prompt = f"""You are maintaining a running summary of a fantasy role-playing game session.

        Update the existing summary by incorporating the events from the latest round.

        Here is the previous summary:
        {context}

        Here is the most recent round:
        {messages_to_summarize}

        Please provide an updated summary that preserves all relevant details and remains consistent in tone and style."""

        message = summarizer_llm.invoke(prompt)

        return {"context":message,"summarized_messages":message}


    def summarize_condition(self,state:State):
        if self.round_counter >= self.max_seen_rounds:
            
            return "summarize"
        else:
            return "continue"

    def looper(self,state: State):
        print("chatbot stage")
        print(state["messages"][-3*self.max_seen_rounds:])
        message = llm.invoke(state["messages"][-3*self.max_seen_rounds:])

        return {"messages": [message]}
# png_data = LoopGraph().graph.get_graph().draw_mermaid_png()

# # PNG'yi dosyaya yaz:
# with open("graph_png/loop_graph_v8.png", "wb") as f:
#     f.write(png_data)

#pseudocode:

# after running for 10 rounds
# Summarize and store the first 5 rounds in the RAG.
# don't send the first 5 rounds into LLM anymore

# after running for another 5 rounds
# summarize the 5 rounds from the last summarization, along with the summarizaiton
# store them in the rag too
# don't erase the messages for explainable ai

# LLM doesn't have to see more than 20 previous messages
# but it will have different seeing,


#second pseudocode

# after running 8 rounds
# summarize first two into constant amount of tokens and store them in rag
# story, characters json, rules, rag results of the most relevant 3 results, summary will be provided in each round
# after that, summarize and every round then store in rag along with the previous summarization