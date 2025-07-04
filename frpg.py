from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

import streamlit as st

from dotenv import load_dotenv
from static_objects import game,first_prompt,tasks

from creation_graph import FullGraph
from loop_graph import LoopGraph

import os

import pickle

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

load_dotenv()

full_text = ""
    
placeholder = st.empty()  # akış için boş bir alan

def print_to_streamlit(events, full_text):
    """
    events: LangGraph akışından gelen event listesi
    full_text: Streamlit alanında biriken metni tutmak için
    """
    previous_id = ""
    for event in events:

        if "messages" in event:
            msg = event["messages"][-1]

            if msg.id == previous_id: 
                continue
            
            if isinstance(msg, HumanMessage):
                full_text += f"\n**Human:** {msg.content}\n"
                
            elif isinstance(msg, AIMessage):
                full_text += f"\n**AI:** {msg.content}\n"
                
            else: 
                continue    
            # Güncellenen tüm metni placeholder'a yaz


            previous_id = msg.id
            placeholder.write(full_text)
    
    return full_text

if "graph" not in st.session_state:

    full_graph = FullGraph()
    config = full_graph.config
    graph = full_graph.graph
    
    _loop_graph = LoopGraph()
    loop_config = _loop_graph.config
    loop_graph = _loop_graph.graph

    
    
    for i in range(len(tasks)):
        task = tasks[i]
        schema_no = i

        events = graph.stream({"messages": [{"role": "user", "content":task},],"current_task":task,"current_schema_no":schema_no},config,stream_mode="values")

        full_text = print_to_streamlit(events,full_text)
    

    st.write(game.characters)
    st.write(game.story)
    st.write(game.rules or "Game rules weren't created!")

    with open('game.pkl', 'wb') as f:

        pickle.dump(game,f)

    st.session_state.graph = graph
    st.session_state.config = config
    st.session_state.full_text = full_text

else:
    
    graph = st.session_state.graph
    config = st.session_state.config
    full_text = st.session_state.full_text

if 'input_text' not in st.session_state:
    st.session_state.input_text = ''

def submit():
    st.session_state.input_text = st.session_state.widget
    st.session_state.widget = ""

st.text_input('Something', key='widget', on_change=submit)

if st.session_state.input_text:

    events = graph.stream(
        {"messages": [{"role": "user", "content": st.session_state.input_text}]},
        config,
        stream_mode="values",
    )

    st.session_state.full_text = print_to_streamlit(events,full_text)

    
