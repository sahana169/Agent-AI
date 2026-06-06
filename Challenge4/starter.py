# Challenge 4 - Full Agent: Tools + Persistent Memory
# Combines Challenge 2 (tools) and Challenge 3 (memory) into one agent.
#
# Features:
#   - Calculator, Weather, Age Calculator tools (@tool)
#   - Persistent memory via Mem0 + FAISS (fully local, no API keys)
#   - Ollama llama3.2:3b for both chat and memory extraction
#   - nomic-embed-text for vector embeddings
#   - Interactive chat loop that remembers you across sessions
#
# ─── SETUP (run once) ─────────────────────────────────────────────────────────
#   pip install "strands-agents[ollama]" mem0ai faiss-cpu
#   ollama pull llama3.2:3b
#   ollama pull nomic-embed-text
# ──────────────────────────────────────────────────────────────────────────────

import os
import shutil
from strands import Agent, tool
from strands.models.ollama import OllamaModel
from mem0 import Memory

# =============================================================================
# SECTION 1: TOOLS
# @tool turns a Python function into something the agent can call automatically.
# The agent reads the docstring to decide when to use each tool.
# Type hints are required so the agent knows what arguments to pass.
# =============================================================================

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a math expression and return the result.
    Use this for any arithmetic: addition, subtraction, multiplication, division.
    Examples: '10 * 5 + 3', '100 / 4', '2 ** 8'
    """
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Returns temperature and conditions for the requested location.
    """
    # Simulated data — swap for a real API (e.g. OpenWeatherMap) if needed
    data = {
        "new york": "22°C, Partly Cloudy",
        "london":   "15°C, Rainy",
        "tokyo":    "28°C, Sunny",
        "sydney":   "18°C, Windy",
        "paris":    "20°C, Clear skies",
        "dubai":    "38°C, Sunny and Hot",
        "mumbai":   "32°C, Humid",
        "chennai":  "35°C, Hot and Humid",
    }
    result = data.get(city.lower().strip())
    if result:
        return f"Weather in {city.title()}: {result}"
    return f"No data for '{city}'. Available: {', '.join(c.title() for c in data)}"


@tool
def calculate_age(birth_year: int, birth_month: int, birth_day: int) -> str:
    """
    Calculate a person's current age from their date of birth.
    Provide birth_year, birth_month (1-12), and birth_day (1-31).
    Example: birth_year=1995, birth_month=6, birth_day=15
    """
    from datetime import date
    try:
        today    = date.today()
        birthday = date(birth_year, birth_month, birth_day)
        age = today.year - birthday.year - (
            (today.month, today.day) < (birthday.month, birthday.day)
        )
        return f"Age: {age} years old (born {birthday.strftime('%B %d, %Y')})"
    except ValueError as e:
        return f"Invalid date: {e}"


# =============================================================================
# SECTION 2: MEMORY SETUP (Mem0 + FAISS)
#
# Mem0 is a memory layer that:
#   - Extracts facts from conversation ("My name is X" → stores "User's name is X")
#   - Saves them as vectors in FAISS on disk (./faiss_memory/)
#   - Lets you search for relevant memories by semantic similarity
#
# We use Ollama for both the LLM (fact extraction) and embedder (vectorization).
# EMBED_DIMS must match what nomic-embed-text actually outputs (768).
# =============================================================================
FAISS_PATH = "./faiss_memory"
COLLECTION = "full_agent_memory"
EMBED_DIMS = 768  # nomic-embed-text output dimensions

# Auto-fix stale index: if an old index has wrong dims, delete and rebuild
index_file = os.path.join(FAISS_PATH, f"{COLLECTION}.faiss")
if os.path.exists(index_file):
    import faiss as _faiss
    _idx = _faiss.read_index(index_file)
    if _idx.d != EMBED_DIMS:
        print(f"[Info] Stale FAISS index (dims={_idx.d}). Rebuilding with {EMBED_DIMS}...")
        shutil.rmtree(FAISS_PATH, ignore_errors=True)

mem0_config = {
    "vector_store": {
        "provider": "faiss",
        "config": {
            "collection_name": COLLECTION,
            "path": FAISS_PATH,
            "distance_strategy": "cosine",      # Best for semantic text similarity
            "embedding_model_dims": EMBED_DIMS, # Must match embedder output
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0,                   # Low temp = precise fact extraction
            "max_tokens": 1000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
        },
    },
}

# Creates ./faiss_memory/ on first run; reloads it on subsequent runs
memory = Memory.from_config(mem0_config)

# =============================================================================
# SECTION 3: STRANDS AGENT SETUP
#
# The Agent gets all three tools in its tools=[] list.
# When you ask "what is 25 * 4?", it automatically calls calculator.
# When you ask "weather in Tokyo?", it calls get_weather — no manual routing needed.
# Memory is injected manually into each prompt (see Section 5).
# =============================================================================
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are a helpful personal assistant with tools and memory. "
        "Use the calculator tool for math, get_weather for weather queries, "
        "and calculate_age for age questions. "
        "You will receive relevant memories at the start of each message — "
        "use them to personalize your responses. "
        "Acknowledge when users share new information about themselves."
    ),
    tools=[calculator, get_weather, calculate_age],
)

# User ID links all memories to this person in Mem0
USER_ID = "user_001"

# =============================================================================
# SECTION 4: MEMORY HELPER FUNCTIONS
# =============================================================================

def save_to_memory(user_message: str, agent_response: str):
    """Save a conversation turn. Mem0 auto-extracts facts and stores them."""
    messages = [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": agent_response},
    ]
    memory.add(messages, user_id=USER_ID)


def get_relevant_memories(query: str) -> str:
    """Search for memories relevant to the current query. Returns formatted string."""
    results = memory.search(query, filters={"user_id": USER_ID}, limit=5)
    items = results.get("results", [])
    if not items:
        return ""
    lines = [f"- {m['memory']}" for m in items]
    return "Relevant memories:\n" + "\n".join(lines)


# =============================================================================
# SECTION 5: INTERACTIVE CHAT LOOP
#
# Each turn follows this flow:
#   1. Fetch relevant memories from FAISS
#   2. Prepend them to the user's message
#   3. Send the combined prompt to the Strands agent
#   4. Save the turn to memory for future recall
# =============================================================================
print("Full Agent ready! I have tools and memory.")
print("Try: 'My name is Sahana', 'What is 144 / 12?', 'Weather in Tokyo'")
print("     'How old is someone born on 1995-06-15?', 'What is my name?'")
print("Type 'memories' to view stored memories. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ("exit", "quit"):
        print("Goodbye! Your memories are saved for next time.")
        break

    # Show all stored memories on demand
    if user_input.lower() == "memories":
        all_mem = memory.get_all(filters={"user_id": USER_ID})
        items = all_mem.get("results", [])
        if items:
            print("\n--- Stored Memories ---")
            for i, m in enumerate(items, 1):
                print(f"{i}. {m['memory']}")
            print("-----------------------\n")
        else:
            print("No memories stored yet.\n")
        continue

    # Step 1: Retrieve relevant past memories
    memory_context = get_relevant_memories(user_input)

    # Step 2: Build prompt — memories prepended so agent is aware of them
    prompt = f"{memory_context}\n\nUser message: {user_input}" if memory_context else user_input

    # Step 3: Agent responds (auto-calls tools if needed)
    print("Assistant: ", end="", flush=True)
    response = agent(prompt)
    print()

    # Step 4: Save turn to memory
    response_text = str(response) if not isinstance(response, str) else response
    save_to_memory(user_input, response_text)
