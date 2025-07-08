from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

import streamlit as st

from dotenv import load_dotenv
from static_objects import game,first_prompt,tasks,theme

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

            if isinstance(msg, AIMessage):
                full_text += f"\n**AI:** {msg.content}\n"
                

            # Güncellenen tüm metni placeholder'a yaz


            previous_id = msg.id
            placeholder.write(full_text)
    
    return full_text

def print_text_to_streamlit(text, full_text):
    """
    tetx: LangGraph akışına gönderilen text
    full_text: Streamlit alanında biriken metni tutmak için
    """
    full_text += f"\n**Human:** {text}\n"
    
    placeholder.write(full_text)
    
    return full_text

def create_game(graph,config,full_text,save_created=False,override_save = False):

    temp = f"The details about my game are provided, please comply these and especially the game theme:{theme}"

    for i in range(len(tasks)):
        
        schema = {"story is: ":game.story,"rules are:":game.rules,"characters are:":game.characters} # schema is provided here so that the values inside are up to date

        task = tasks[i]
        schema_no = i

        if i!=0:
            key = list(schema.keys())[i-1]
            temp+= (key+str(schema[key]))
            task+=temp
        
        events = graph.stream({"messages": [{"role": "user", "content":task},],"current_task":task,"current_schema_no":schema_no},config,stream_mode="values")

        full_text = print_to_streamlit(events,full_text)
    

    st.write(game.characters)
    st.write(game.story)
    st.write(game.rules or "Game rules weren't created!")

    if save_created:
        name = "game_1"

        if not override_save:
            i = 1
            while os.path.exists(name):
                lst = name.split(".")
                name = lst[0]+"_"+str(i)+".pkl"

        with open(f'{name}.pkl', 'wb') as f:

            pickle.dump(game,f)

def define_non_player(list_of_players):
    for i in game.characters.keys():
        if i in list_of_players:
            game.characters[i]["character_type"] = "player"
        else:
            game.characters[i]["character_type"] = "npc"
        
def start_game(graph,config,full_text):

    game.main_character = list(game.characters.keys())[-1]
    main_character = game.main_character
    
    define_non_player([main_character])

    content = f"""My character is {main_character}, other characters are NPCs. 
    You will play the rounds of NPCs and won't give the decisions to me. The round order for me and other characters is in this order: {list(game.characters.keys())} , please obey this order. 
    Based on the provided details, first tell our current situation, explain the story and continue the game in round order by playing the NPCs. 
    The story should go like this, start from the start, you can manipulate the story if needed:{game.story}. 
    """

    invoke_loop(graph,config,full_text,content=content)

def invoke_loop(graph,config,full_text,content = None):

    content = st.session_state.input_text if not content else content

    
    full_text = print_text_to_streamlit(content,full_text)


    content+=f"The rules are {game.rules}. The characters and their inventories stats and other details are : {game.characters}"

    events = graph.stream(
        {"messages": [{"role": "user", "content": content}]},
        config,
        stream_mode="values",
    )

    st.session_state.full_text = print_to_streamlit(events,full_text)


if "full_graph" not in st.session_state:

    full_graph = FullGraph()
    full_config = full_graph.config
    full_graph = full_graph.graph
    
    _loop_graph = LoopGraph()
    loop_config = _loop_graph.config
    loop_graph = _loop_graph.graph
    
    

    #create_game(full_graph,full_config,full_text,save_created=True,override_save=True)

    start_game(loop_graph,loop_config,full_text)

    st.session_state.full_graph = full_graph
    st.session_state.full_config = full_config
    st.session_state.loop_graph = loop_graph
    st.session_state.loop_config = loop_config
    st.session_state.full_text = full_text

else:
    
    full_graph = st.session_state.full_graph
    full_config = st.session_state.full_config
    loop_graph = st.session_state.loop_graph
    loop_config = st.session_state.loop_config
    full_text = st.session_state.full_text

if 'input_text' not in st.session_state:
    st.session_state.input_text = ''

def submit():
    st.session_state.input_text = st.session_state.widget
    st.session_state.widget = ""

st.text_input('Something', key='widget', on_change=submit)

if st.session_state.input_text:

    invoke_loop(loop_graph,loop_config,full_text)
