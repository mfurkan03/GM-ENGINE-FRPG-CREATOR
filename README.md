# 🧙 FRPG - Fantasy Role Playing Game Engine (LangGraph + LLM)

This is a text-based fantasy role-playing game powered by **LangGraph**, **LangChain tools**, and **LLMs (Language Models)**. Players can interact with a dynamic world, talk to NPCs, make decisions, roll dice, and manage inventory — all within an AI-powered game master system.

---

## 🎮 Features

- 🗺️ **Story Engine**: Dynamic fantasy storyline built turn-by-turn.
- 🎭 **NPCs with Personality**: Each NPC has stats, traits, and motivations.
- 🧠 **LLM-Powered GM**: A language model acts as the game master.
- 🎲 **Dice System**: Roll d20s and other dice with classic RPG mechanics.
- 🎒 **Inventory & Currency Tracking**: Items and money are managed automatically.
- 🧪 **Tool-Triggered Actions**: Characters gain/lose items, money, or trigger combat via LangChain tools.

---

## 🛠️ Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — Graph-based orchestration for LLM workflows.
- [LangChain](https://www.langchain.com/) — Tool calling, memory, and state management.
- `ChatGroq` or `ChatOllama` — Model connectors (e.g., Mistral, Qwen, GPT).
- `Streamlit` — Frontend interface for playing the game.
- Python 3.10+

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/frpg.git
cd frpg
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Set API Keys
```bash
GROQ_API_KEY=your-groq-key
```
### 4. Run the Game
```bash
streamlit run frpg.py
```
### 📦 Project Structure
```graphql
frpg/
├── frpg.py                     # Main Streamlit interface
├── game_functions.py           # Tool functions (inventory, dice, money)
├── loop_graph.py               # LangGraph game loop logic and loop graph object
├── creation_graph.py           # LangGraph game creation logic and creation graph object
├── static_objects.py           # Global game state (characters, story, rules) and some of the prompts
├── requirements.txt
└── README.md
```

📌 Notes
Inventory is passed to the LLM as a HumanMessage each turn — old inventory messages are pruned for token efficiency.

Story turns are formatted with round headers, dialogues, events, and summaries.

The system supports @tool decorated functions via LangChain.

📜 License
MIT License © 2025

🙋‍♂️ Acknowledgments
This project uses:

LangChain & LangGraph

Qwen/Gemma/Llama LLMs via Groq or Ollama

Inspiration from classic tabletop RPGs
