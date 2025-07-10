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

prompts = []
system_message = SystemMessage("You are a professional FRPG game designer. Don't answer me, only do the job I tell. Don't use tools too much for formatting etc. When calling functions, ensure the JSON uses lowercase true/false for booleans. Important: Be original,creative,unique!")

theme = "cyberpunk"

rule_format = f"""for my frpg game, my theme is {theme}:  There are Players and a Game Master (GM). Ability checks use 2d10 (two ten-sided dice), summing the results. 
To succeed, you must meet or exceed the Difficulty Rating (DR) set by the GM: 
Easy: DR 12 Medium: DR 16 Hard: DR 20 Very Hard: DR 24 If you have relevant expertise, 
add +3; if you’re a master, add +5. Character Creation Your character has 3 core attributes: 
Body : Physical strength, endurance Reflexes : Speed, agility, aim Mind : Intelligence, cyber skills, 
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

Covering key gameplay aspects, including combat, inventory, and interaction with NPCs. No rule about movement!

Once the rules are ready, don't forget to store them in the database using define_rules()!""",


"""Create the main characters:

Write a short description of each character, including their personality or backstory.

Assign stats for each character in dictionary format (e.g., {"strength": 8, "intelligence": 6}, don't use abbreviations like BOD or MND etc.) Also give money and HP to the characters.

Add each NPC to the game using the tool: 'add_or_change_character' . But don't create the inventories yet! The user will later select which character to play. """,


"""
For each NPC:

Create appropriate items or weapons.

Write a description or explanation for each item.

Assign stats to items if applicable (as a dictionary).

Add the items to the corresponding NPC’s inventory using add_or_change_item_to_character_inventory."""]

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

def sanitize_message_content(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Args:
        messages: List of BaseMessage objects to sanitize
        
    Returns:
        List[BaseMessage]: Sanitized message list with empty text entries removed
    """
    sanitized_messages = []

    for msg in messages:
        # Create a copy of the message to avoid modifying the original
        if isinstance(msg, AIMessage) or isinstance(msg, AIMessageChunk):
            # Handle content that is a list of content items (dict with type/text)
            if isinstance(msg.content, list):
                # Filter out content items with empty text
                non_empty_content = [
                    item for item in msg.content 
                    if not (item.get('type') == 'text' and (item.get('text', '') == '' or item.get('text') is None))
                ]
                
                # If we have non-empty content, create a new message with it
                if non_empty_content:
                    new_msg = AIMessage(
                        content=non_empty_content,
                        additional_kwargs=msg.additional_kwargs,
                        response_metadata=msg.response_metadata,
                        id=msg.id,
                        tool_calls=getattr(msg, 'tool_calls', None),
                        usage_metadata=getattr(msg, 'usage_metadata', None)
                    )
                    sanitized_messages.append(new_msg)
                else:
                    # If all content was empty and removed, create a simple non-empty message
                    # to maintain the conversation flow
                    new_msg = AIMessage(
                        content=[{'type': 'text', 'text': '[No content]'}],
                        additional_kwargs=msg.additional_kwargs,
                        response_metadata=msg.response_metadata,
                        id=msg.id,
                        tool_calls=getattr(msg, 'tool_calls', None),
                        usage_metadata=getattr(msg, 'usage_metadata', None)
                    )
                    sanitized_messages.append(new_msg)
            else:
                # Handle string content - ensure it's not empty
                if not msg.content:
                    content = "[No content]"
                # Handle case when content is a string representation of a list with empty text
                elif isinstance(msg.content, str) and msg.content.startswith("[{'type': 'text', 'text': ") and ("''" in msg.content or '""' in msg.content):
                    content = "[No content]"
                elif isinstance(msg.content, list) and len(msg.content) == 1:
                    item = msg.content[0]
                    if isinstance(item, dict) and item.get('type') == 'text' and not item.get('text', '').strip():
                        content = "[No content]"
                    else:
                        content = msg.content
                else:
                    content = msg.content
                
                new_msg = AIMessage(
                    content=[{'type': 'text', 'text': content}] if isinstance(content, str) else content,
                    additional_kwargs=msg.additional_kwargs,
                    response_metadata=msg.response_metadata,
                    id=msg.id,
                    model_fields_set=msg.model_fields_set,
                    tool_calls=getattr(msg, 'tool_calls', None),
                    usage_metadata=getattr(msg, 'usage_metadata', None)
                )
                sanitized_messages.append(new_msg)
        else:
            # For non-AI messages, keep as is
            sanitized_messages.append(msg)

    return sanitized_messages