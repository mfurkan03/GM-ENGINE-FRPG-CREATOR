from langchain_core.tools import InjectedToolCallId, tool
from typing import Annotated
from static_objects import game

@tool
def add_character(
    name: Annotated[str, "Unique character name"],
    details: Annotated[str, "Character description, personality, appearance, or backstory"],
    money: Annotated[int, "Initial in-game currency for the character"],
    stats: Annotated[dict, "Character's stats as a dictionary, e.g., {'strength': 8}"]
):
    """
    Add a new character to the game.

    Parameters:
        name (str): The name of the new character. Must be unique within the game.
        details (str): A description of the character, such as personality traits, appearance, or backstory.
        money (int): The amount of in-game currency the character starts with.
        stats (dict): A dictionary representing the character's statistics or attributes. Example format:
                    {
                        "strength": 8,
                        "dexterity": 6,
                        "intelligence": 7
                    }

    Behavior:
        - Creates a new character entry in the global 'characters' dictionary.
        - Initializes the character's inventory as an empty list. Populating the inventory should be done separately using other tools.
        - The character's information is stored in a dictionary with the following structure:
            {
                "name": <name>,
                "details": <details>,
                "stats": <stats>,
                "money": <money>,
                "inventory": []
            }

    Important Notes:
        - When calling this tool with an LLM or manually, ensure 'stats' is a properly formatted dictionary. Example:
            stats = {"strength": 10, "agility": 8}
        - If the 'name' key already exists in the global 'characters' dictionary, this will overwrite the existing character without warning.
        - Consider validating unique character names or confirming overwrites in production code.

    Example usage:
        add_character(
            name="Artemis",
            details="A stealthy ranger with a mysterious past.",
            money=100,
            stats={"dexterity": 9, "perception": 8, "luck": 5}
        )
    """
    template = {
        "name": name,
        "details": details,
        "stats": stats,
        "money": money,
        "inventory": []
    }
    game.characters[name.lower()] = template

@tool
def add_item_to_character_inventory(
    character_name: Annotated[str, "Name of the character to receive the item"],
    is_weapon: Annotated[bool, "True if the item is a weapon, False otherwise"],
    item_name: Annotated[str, "Name of the item to add"],
    details: Annotated[str, "Description of the item"],
    stats: Annotated[dict, "Stats dictionary for weapons (e.g., {'damage': 10}); can be empty for regular items"],
    value: Annotated[int, "Value of the item in in-game currency"]
):
    """
    Adds a weapon or an item to the specified character's inventory.

    Parameters:
        character_name (str): The name of the character to whom the item will be added.
        is_weapon (bool): True if the item is a weapon; False if it is a regular item.
        item_name (str): The name of the item to add.
        details (str): A text description providing details about the item.
        stats (dict): A dictionary of stats for the weapon. Example format:
                    {
                        "damage": 10,
                        "speed": 5,
                        "range": 3
                    }
                    Note: If is_weapon is False, this field can be an empty dictionary or omitted.
        value (int): The value or cost of the item in in-game currency.

    Behavior:
        - If is_weapon is True, the function creates an inventory item dictionary with keys:
            - "name": item_name
            - "details": details
            - "stats": stats
            - "value": value
        - If is_weapon is False, the function creates an inventory item dictionary with keys:
            - "name": item_name
            - "details": details
            - "value": value
        - The constructed item dictionary is appended to the character's "inventory" list.

    Important Notes:
        - Ensure the character exists in the global 'characters' dictionary before calling this function.
        - When adding a weapon, the 'stats' argument **must** be a dictionary containing relevant attributes.
        - If you use an LLM to call this tool, explicitly specify the 'stats' field as a dictionary. Example:
            stats = {"damage": 15, "crit_chance": 0.2}
        - If stats is missing or incorrectly formatted, the system may throw an error or produce unintended behavior.

    Example usage:
        add_item_to_character_inventory(
            character_name="Alice",
            is_weapon=True,
            item_name="Excalibur",
            details="A legendary sword with immense power.",
            stats={"damage": 100, "durability": 80},
            value=5000
        )
    """
    if is_weapon:
        template = {
            "name": item_name,
            "details": details,
            "stats": stats,
            "value": value
        }
    else:
        template = {
            "name": item_name,
            "details": details,
            "value": value
        }

    game.characters[character_name.lower()]["inventory"].append(template)
    
@tool
def delete_item_from_character_inventory(character_name: Annotated[str, "Name of the character whose inventory will be modified"],
    item_name: Annotated[str, "Name of the item to delete from the character's inventory"]):
    """
    Erases the specified item from the specified character's inventory.
    """
    # Check if the character exists.
    char_key = character_name.lower()
    if char_key not in game.characters:
        raise KeyError(f"Character couldn't be found: {character_name}")

    inventory = game.characters[char_key].get("inventory")
    if inventory is None:
        raise KeyError(f"{character_name} does not have an inventory.")

    # Find the item and erase it from the inventory.
    try:
        inventory.remove(item_name)
    except ValueError:
        raise ValueError(f"{item_name} isn't in the inventory.")

@tool
def define_rules(rules: Annotated[str, "Game rules written in plain language to enforce gameplay compliance"]):
    "Add game rules in a string so that another LLM can read these and decide if the actions are comply with these rules."
    game.rules= rules
        
    
