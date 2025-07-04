from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

import streamlit as st

from dotenv import load_dotenv
from static_objects import game,first_prompt,tasks

from construct_graph import FullGraph

import os

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

load_dotenv()

full_text = ""


if "graph" not in st.session_state:

    full_graph = FullGraph()
    config = full_graph.config
    graph = full_graph.graph

    placeholder = st.empty()  # akış için boş bir alan


    def print_to_streamlit(events,full_text):
                    # biriken metni tutmak için

        for event in events:
            if "messages" in event and isinstance(event["messages"][-1], HumanMessage) or isinstance(event["messages"][-1], AIMessage):
                full_text += str(event["messages"][-1].content)    # yeni token'ı ekle
                placeholder.write(full_text)  # aynı alanda güncelle
                
        return full_text
    
    for i in range(len(tasks)):
        task = tasks[i]
        schema_no = i

        events = graph.stream({"messages": [{"role": "user", "content":task},],"current_task":task,"current_schema_no":schema_no},config,stream_mode="values")

        full_text = print_to_streamlit(events,full_text)
    

    st.write(game.characters)
    st.write(game.story)
    st.write(game.rules or "Game rules weren't created!")

    st.session_state.graph = graph
    st.session_state.config = config
    st.session_state.full_text = full_text
else:
    
    graph = st.session_state.graph
    config = st.session_state.config
    full_text = st.session_state.full_text


input_text=st.text_input("...")

if input_text:

    events = graph.stream(
        {"messages": [{"role": "user", "content": input_text}]},
        config,
        stream_mode="values",
    )

    st.session_state.full_text = print_to_streamlit(events,full_text)
    st.write(graph.get_state_history(config))
