
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

import streamlit as st


from static_objects import game,generate_tasks,new_game,load_game

from creation_graph import FullGraph
from loop_graph import LoopGraph
import requests
from PIL import Image
from io import BytesIO
import os
import pickle

from playsound import playsound
import base64
import threading
import time

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

    
placeholder = st.empty()  # akış için boş bir alan

def print_to_streamlit(events, full_text):
    """
    events: LangGraph akışından gelen event listesi
    full_text: Streamlit alanında biriken metni tutmak için
    """
    previous_id = ""
    new_msg = ""
    for event in events:

        if "messages" in event:
            msg = event["messages"][-1]

            if msg.id == previous_id or msg.content == "": 
                continue
            
            if isinstance(msg, AIMessage):
                new_msg += f"\n**AI:** {msg.content}\n"
                

            # Güncellenen tüm metni placeholder'a yaz

    full_text+=new_msg
    previous_id = msg.id
    placeholder.write(full_text)

    return full_text,new_msg

def print_text_to_streamlit(text, full_text,new_text = True):
    """
    tetx: LangGraph akışına gönderilen text
    full_text: Streamlit alanında biriken metni tutmak için
    """
    if new_text:
        full_text += f"\n**Human:** {text}\n"
    
    placeholder.write(full_text)
    
    return full_text

def create_game(theme, graph,config,full_text,save_created=False,override_save = True):

    new_game()

    temp = f"The details about my game are provided, please comply these and especially the game theme:{theme} "
    tasks = generate_tasks(theme)
    for i in range(len(tasks)):
        
        schema = {"story is: ":game.story,"rules are:":game.rules,"characters are:":game.characters} # schema is provided here so that the values inside are up to date

        task = tasks[i]
        schema_no = i

        if i!=0:
            key = list(schema.keys())[i-1]
            temp+= (key+str(schema[key]))
            task+=temp
        events = graph.stream({"messages": [{"role": "user", "content":task},],"current_task":task,"current_schema_no":schema_no},config,stream_mode="values")

        full_text,new_msg = print_to_streamlit(events,full_text)
    

    st.write(game.characters)
    st.write(game.story)
    st.write(game.rules or "Game rules weren't created!")

    if save_created:
        name = "game"

        if not override_save:
            i = 1
            while os.path.exists(f'{name}.pkl'):
                lst = name.split(".")
                name = lst[0]+"_"+str(i)+".pkl"

        with open(f'{name}.pkl', 'wb') as f:

            pickle.dump(game,f)

        return f'{name}.pkl'

def define_non_player(list_of_players):

    for i in game.characters.keys():
        if i in list_of_players:
            game.characters[i]["character_type"] = "player"
        else:
            game.characters[i]["character_type"] = "npc"


def start_game(graph_object,config,full_text):

    game.main_character = list(game.characters.keys())[-1]
    main_character = game.main_character
    
    define_non_player([main_character])
    
    return invoke_loop(graph_object,config,full_text)

def play_audio(audio):
    playsound(audio)

def invoke_loop(graph_object,config,full_text,content = None):

    graph = graph_object.graph

    content = st.session_state.input_text if not content else content

    st.session_state.input_text = ""
    
    full_text = print_text_to_streamlit(content,full_text)

    events = graph.stream(
        {"messages": [{"role": "user", "content": content}]},
        config,
        stream_mode="values",
    )
    
    graph_object.round_counter+=1

    full_text,new_msg = print_to_streamlit(events,full_text)

    chars = []
    
    for char in game.characters.values():
        chars.append(str(char))

    url = 'https://a564a680702b.ngrok-free.app/generate'


    payload = {
    "narrative": new_msg,
    "characters": chars,
    "previous_image_url": None,
    "previous_image_style": st.session_state.previous_image_style
    }
    headers = {
        "ngrok-skip-browser-warning": "69420"
    }
    image_url = None
    
    response = requests.post(url, json=payload, headers=headers)
    print(response.json())
    if response.status_code == 200:
        image_url = response.json()["image_url"]
        music = response.json()["music_url"]

        st.session_state.previous_image_url = image_url
        print(st.session_state.previous_image_url+"aslkdmas")

        st.session_state.previous_image_style = response.json()["new_image_style"]


    return full_text

if "full_graph" not in st.session_state:

    full_graph = FullGraph()
    full_config = full_graph.config
    full_graph = full_graph.graph
    
    loop_graph = LoopGraph()
    loop_config = loop_graph.config
    
    st.session_state.previous_image_url = None
    st.session_state.previous_image_style = None
    
    full_text = ""
    st.session_state.input_text = ''

    # pkl = create_game("pokemon",full_graph,full_config,full_text,save_created=True,override_save=False)
    # st.session_state.created_game_pkl = pkl

    load_game("game_1.pkl")

    st.sidebar.json(game.characters)

    full_text = start_game(loop_graph,loop_config,full_text)


    st.session_state.full_graph = full_graph
    st.session_state.full_config = full_config
    st.session_state.loop_graph = loop_graph
    st.session_state.loop_config = loop_config
    st.session_state.full_text = full_text
    st.session_state.game = game
    st.session_state.refreshed = False    
else:

    full_graph = st.session_state.full_graph
    full_config = st.session_state.full_config
    loop_graph = st.session_state.loop_graph
    loop_config = st.session_state.loop_config
    full_text = st.session_state.full_text
    game = st.session_state.game

    

def submit():
    st.session_state.input_text = st.session_state.widget
    st.session_state.widget = ""

if "previous_image_url" in st.session_state:
    print("b")
    try:
        print("c")
        print(st.session_state.previous_image_url)
        res = requests.get(st.session_state.previous_image_url)
        img = Image.open(BytesIO(res.content))
        st.image(img)
    except Exception as e:
        st.error("Görsel yüklenemedi.")
else:
    st.warning("Görsel oluşturulamadı.")


def write_refreshed():
    st.session_state.refreshed = True

if st.session_state.refreshed:
    print_text_to_streamlit("",st.session_state.full_text,False)
    st.sidebar.json(game.characters)
    st.session_state.refreshed = False

st.text_input('Something', key='widget', on_change=submit)
st.button("refresh",key = 'button1',on_click=write_refreshed)

if st.session_state.input_text and st.session_state.input_text!="":    

    st.session_state.full_text = invoke_loop(loop_graph,loop_config,full_text)
    st.sidebar.json(game.characters)
    
