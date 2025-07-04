from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain.prompts import ChatPromptTemplate

class GameContext:
    def __init__(self):
        self.characters = {}
        self.rules = ""

game = GameContext()

prompts = []
system_message = SystemMessage("You are a professional FRPG game designer.")

theme = "cyberpunk"

rule_format = f"rules should be in this format for my frpg game, my theme is {theme}:  There are Players and a Game Master (GM). Ability checks use 2d10 (two ten-sided dice), summing the results. To succeed, you must meet or exceed the Difficulty Rating (DR) set by the GM: Easy: DR 12 Medium: DR 16 Hard: DR 20 Very Hard: DR 24 If you have relevant expertise, add +3; if you’re a master, add +5. Character Creation Your character has 3 core attributes: Body (BOD): Physical strength, endurance Reflexes (REF): Speed, agility, aim Mind (MND): Intelligence, cyber skills, analysis At character creation, distribute 4, 3, and 1 points among these attributes (total of 8 points). Additionally, choose: One expertise/role (e.g., Netrunner, Street Samurai, Mechanic). One motivation (e.g., Cause chaos, Save family, Seek knowledge). One flaw (e.g., Paranoia, Substance addiction, Ambition).  Combat System For initiative, roll 2d10 + REF, highest goes first. For attacks, roll 2d10 + REF or BOD + expertise bonus, aiming to beat the defense DR (typically 16 + enemy’s armor rating). Damage: Light weapons: 1d8 Medium-caliber/one-handed melee weapons: 1d10 Heavy weapons: 1d12 The GM can adjust damage based on narrative context and weapon type.  Health Each character has 15 + BOD points as Hit Points (HP). When HP drops to zero, the character becomes critically injured and rolls a death die: 1d10 → below 5 means death; 5 or higher means unconscious but alive.  Cyber Powers & Hacking Mechanics To use cyber abilities or hack, make a MND-based check: 2d10 + MND + cyber expertise bonus ≥ hack DR (DR = 14 + target system’s security level). On failure, the system triggers an alarm, security responds, and the hacker loses their next action. Advancement At the end of each adventure, the GM awards 1–2 advancement points. At 4 advancement points, you can increase an attribute by +1 or gain a new expertise. Acquiring cyber abilities or implants requires GM approval.  Sample Character Name: Neon BOD 3 / REF 4 / MND 1 Expertise: Street Samurai Motivation: Revenge against megacorps Flaw: Uncontrolled anger HP: 18"

tasks = [


f"""Create an outline for the game's story in theme: {theme}.""",


f"""
Write a clear set of game rules. These rules must be in this format{rule_format}:

Simple and unambiguous.

Written in plain language that can be understood and enforced by another LLM for rule-compliance evaluation.

Covering key gameplay aspects, including combat, movement, inventory, and interaction with NPCs.

Once the rules are ready, don't forget to store them in the database using define_rules()!""",


"""Create the main NPC characters:

Write a short description of each NPC, including their personality or backstory.

Assign stats for each NPC in dictionary format (e.g., {"strength": 8, "intelligence": 6}).

Add each NPC to the game using the add_character tool.""",


"""
For each NPC:

Create appropriate items or weapons.

Write a description or explanation for each item.

Assign stats to items if applicable (as a dictionary).

Add the items to the corresponding NPC’s inventory using add_item_to_character_inventory.""",



"""
provide the user with a set of choices to design their own starting character. Include options for appearance, initial stats, and a brief backstory.
"""]

for i in range(len(tasks)):
    
    human_message =tasks[i]

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message,
    ])

    tasks[i] = chat_prompt

first_prompt = ""