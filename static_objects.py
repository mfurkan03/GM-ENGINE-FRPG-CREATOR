from typing import List
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage,BaseMessage,BaseMessageChunk,AIMessageChunk
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
import json

import pickle

class GameContext:
    def __init__(self):
        self.characters = {}
        self.rules = ""
        self.story = ""
    
    def value(self, new_value):
        old_value = self._value
        self._value = new_value
        for callback in self._listeners:
            callback(old_value, new_value)
            
# game = GameContext()

with open('game_1.pkl', 'rb') as file:
    game = pickle.load(file)


def new_game():
    global game
    game.characters = {}
    game.story = ""
    game.rules = ""

def load_game(game_pkl):
    global game

    with open(game_pkl, 'rb') as file:
        game_2 = pickle.load(file)

    game.characters = game_2.characters
    game.story = game_2.story
    game.rules = game_2.rules

theme = "medieval_dynasty"

prompts = []

system_message = SystemMessage("You are a professional FRPG game designer. Don't answer me, only do the job I tell. Don't use tools too much for formatting etc. When calling functions, ensure the JSON uses lowercase true/false for booleans. Important: Be original,creative,unique!")




def generate_tasks(theme):

    rule_format = f"""for my frpg game please create rules in this format. Not exactly these but similar, my theme is {theme}:  There are Players and a Game Master (GM). Ability checks use 2d10 (two ten-sided dice), summing the results. 
                To succeed, you must meet or exceed the Difficulty Rating (DR) set by the GM: 
                Easy: DR 12 Medium: DR 16 Hard: DR 20 Very Hard: DR 24 If you have relevant expertise, 
                add +3; if you’re a master, add +5. Character Creation Your character has 3 core attributes: 
                Body : Physical strength, endurance Reflexes : Speed, agility, aim Mind : Intelligence, cyber skills, (you can change the name of the stats based on the theme)
                analysis At character creation, distribute 4, 3, and 1 points among these attributes (total of 8 points). 
                Additionally, choose: One expertise/role (e.g., Netrunner, Street Samurai, Mechanic). 
                One motivation (e.g., Cause chaos, Save family, Seek knowledge). 
                One flaw (e.g., Paranoia, Substance addiction, Ambition).  
                Combat System For initiative, roll 2d10 + REF, highest goes first. 
                For attacks, roll 2d10 + REF or BOD + expertise bonus, aiming to beat the defense DR (typically 16 + enemy’s armor rating). 
                Damage: Light weapons: 1d8 Medium-caliber/one-handed melee weapons: 1d10 Heavy weapons: 1d12 
                The GM can adjust damage based on narrative context and weapon type.  
                Health Each character has 15 + BOD points as Hit Points (HP). When HP drops to zero, the character becomes critically injured 
                and rolls a death die: 1d10 → below 5 means death; 5 or higher means unconscious but alive.  
                Cyber Powers & Hacking Mechanics To use cyber abilities or hack, make a MND-based check: 2d10 + MND + cyber expertise bonus ≥ hack 
                DR (DR = 14 + target system’s security level). On failure, the system triggers an alarm, security responds, and the hacker loses 
                their next action. Advancement At the end of each adventure, the GM awards 1–2 advancement points. At 4 advancement points, 
                you can increase an attribute by +1 or gain a new expertise. Acquiring cyber abilities or implants requires GM approval.  
                Sample Character Name: Neon BOD 3 / REF 4 / MND 1 Expertise: Street Samurai Motivation: Revenge against megacorps Flaw: 
                Uncontrolled anger HP: 18"""
    
    tasks = [


    f"""Create an outline for the game's story in theme: {theme}. Store this story with define_story tool. Formatting of the text is not that important. Pay attention : I want you to create a starting point for the story, not the whole story.""",


    f"""
    Write a clear set of game rules. These rules must be in this format, but this isn't an exact obligation! Only use main ideas not strict rules! :{rule_format}:

    Simple and unambiguous.

    Written in plain language that can be understood and enforced by another LLM for rule-compliance evaluation.

    Covering key gameplay aspects, including combat, inventory, and interaction with NPCs. No rule about movement! Don't include an inventory capacity rule! Only add that player should be ably to carry humanly carriable things.

    Once the rules are ready, don't forget to store them in the database using define_rules()!""",


    """Create the main characters:
    Create 3 to 5 characters. Don't create more than 5 characters
    Write a short description of each character, including their personality or backstory.

    Assign stats for each character in dictionary format (e.g., {"strength": 8, "intelligence": 6}, don't use abbreviations like BOD or MND etc.) Also give money to the characters but don't give HP since I did not implement it yet.

    Add each NPC to the game using the tool: 'add_or_change_character' . But don't create the inventories yet! The user will later select which character to play.  Remember, don't use add_or_change_item_to_character_inventory() tool in this stage, don't create inventories!

    If you make a mistake in character creation, create and override the character in a clean form using: add_or_change_character() tool
    """,


    """
    For each NPC:

    Create appropriate items or weapons.

    Write a description or explanation for each item.

    Assign stats to items if applicable (as a dictionary).

    Add the items to the corresponding NPC’s inventory using add_or_change_item_to_character_inventory."""]

    return tasks

first_prompt = ""

game_string_format = """### 📖 Story so far
The adventurer entered the dark forest in search of the lost Sigil of Aranth. Strange noises echoed through the trees as they pressed deeper. Suddenly, a group of goblins appeared and surrounded the player.

### 🌀 Round 1 - Player's Turn
- The player shouted: "I draw my sword and attack the nearest goblin!"
- Rolled a 16 on attack.
- The player hit and dealt 7 damage, slaying one goblin instantly.

### 🌀 Round 2 - NPC's Turn
- Goblin Chief snarled: "You will pay for that!"
- The goblins advanced and rolled a 9 on attack.
- The attack missed; the player evaded successfully.

### 🌀 Round 3 - Player's Turn
- The player said: "I cast a fireball at the goblin chief!"
- Rolled a 19 on spellcasting.
- Fireball hit, dealing 12 damage. Goblin Chief is badly burned."""


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list,message_str:str = "messages") -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.str = message_str
        
    def __call__(self, inputs: dict):
        if messages := inputs.get(self.str, []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

def generate_task_prompt():

    main_character =  list(game.characters.keys())[-1]

    task_prompt = f"""My character is {main_character}, other characters are NPCs. 
        You will play the rounds of NPCs and won't give the decisions to me for them.  Only play the npc characters. Don't play the PLAYER characters, 
        I will make every decision about me, don't decide what I will do or say. 
        After I say my input, play the rounds for each npc in order. 
        The round order for me and other characters is in this order: {list(game.characters.keys())}, please obey this order. 
        Based on the provided details, first tell our current situation, explain the story and continue the game in round order by playing the NPCs. 
        
        The story should go like this, start from the start, you can manipulate the story if needed:{game.story}. 

        Don't talk too long, don't talk or describe unnecessarily. Ask for confirmation before any important action from me.

        Write in this format!:
        
        ## **Round 1**
        ### **Character 1's turn**
        ### **Character 2's turn**
        ...
        ### **Player's Character's turn2** : please provide input
        """
    return task_prompt
