# Challenge 3 - Memory Agent using Strands SDK + Mem0 + FAISS + Ollama
#
# How it works:
#   - Mem0 stores memories in a local FAISS vector database (no cloud needed)
#   - Ollama runs the LLM (llama3.2:3b) AND the embedding model (nomic-embed-text)
#   - When you share info like "My name is Sahana", Mem0 saves it
#   - On your next message, relevant memories are fetched and injected into the prompt
#   - The agent then "remembers" who you are across the whole conversation
#
# ─── SETUP COMMANDS (run these once before starting) ──────────────────────────
#   pip install "strands-agents[ollama]" mem0ai faiss-cpu
#   ollama pull llama3.2:3b
#   ollama pull nomic-embed-text
# ──────────────────────────────────────────────────────────────────────────────

import os
import shutil
from strands import Agent
from strands.models.ollama import OllamaModel
from mem0 import Memory

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Stale Index Guard
#
# If a FAISS index exists from a previous run with wrong dimensions (e.g. 1536
# from OpenAI defaults), delete it so a fresh 768-dim index gets created.
# nomic-embed-text produces 768-dimensional vectors.
# ─────────────────────────────────────────────────────────────────────────────
FAISS_PATH  = "./faiss_memory"
COLLECTION  = "agent_memory"
EMBED_DIMS  = 768  # nomic-embed-text output dimensions

index_file = os.path.join(FAISS_PATH, f"{COLLECTION}.faiss")
if os.path.exists(index_file):
    import faiss as _faiss
    _idx = _faiss.read_index(index_file)
    if _idx.d != EMBED_DIMS:
        print(f"[Info] Stale FAISS index (dims={_idx.d}) found. Recreating with {EMBED_DIMS}...")
        shutil.rmtree(FAISS_PATH, ignore_errors=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Mem0 Configuration
#
# Three components:
#   1. vector_store → FAISS (local file-based vector DB, saves to ./faiss_memory/)
#   2. llm          → Ollama llama3.2:3b (Mem0 uses this to extract facts from text)
#   3. embedder     → Ollama nomic-embed-text (converts text to vectors for storage)
# ─────────────────────────────────────────────────────────────────────────────
mem0_config = {
    "vector_store": {
        "provider": "faiss",
        "config": {
            "collection_name": COLLECTION,
            "path": FAISS_PATH,
            "distance_strategy": "cosine",     # Best for semantic similarity
            "embedding_model_dims": EMBED_DIMS, # Critical: must match embedder output
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0,                  # Low temp = consistent fact extraction
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

# Initialize Mem0 — creates ./faiss_memory/ on disk if it doesn't exist
memory = Memory.from_config(mem0_config)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Strands Agent Setup
#
# The Strands agent handles chat via Ollama.
# Mem0 is separate — we fetch relevant memories and inject them into each prompt.
# ─────────────────────────────────────────────────────────────────────────────
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are a helpful personal assistant with memory. "
        "You will be given relevant memories from past conversations at the start of each message. "
        "Use those memories to personalize your responses. "
        "If the user shares new information about themselves, acknowledge it naturally."
    ),
)

# Fixed user ID — links all memories to the same person in Mem0
USER_ID = "user_001"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def save_to_memory(user_message: str, agent_response: str):
    """
    Save a conversation turn to Mem0.
    Mem0 automatically extracts facts (name, preferences, etc.) from the text.
    """
    messages = [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": agent_response},
    ]
    memory.add(messages, user_id=USER_ID)


def get_relevant_memories(query: str) -> str:
    """
    Search Mem0 for memories relevant to the current query.
    Returns a formatted string to inject into the prompt, or empty string if none.
    """
    results = memory.search(query, filters={"user_id": USER_ID}, limit=5)
    memories_list = results.get("results", [])

    if not memories_list:
        return ""

    lines = [f"- {m['memory']}" for m in memories_list]
    return "Relevant memories from past conversations:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Interactive Chat Loop
# ─────────────────────────────────────────────────────────────────────────────
print("Memory Agent ready! I'll remember what you tell me.")
print("Try: 'My name is Sahana' then later ask 'What is my name?'")
print("Type 'memories' to see everything stored. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ("exit", "quit"):
        print("Goodbye! Your memories are saved for next time.")
        break

    # Special command: dump all stored memories
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

    # Step 1: Search for relevant past memories
    memory_context = get_relevant_memories(user_input)

    # Step 2: Prepend memories to prompt so the agent is aware of them
    prompt = f"{memory_context}\n\nUser message: {user_input}" if memory_context else user_input

    # Step 3: Get agent response
    print("Assistant: ", end="", flush=True)
    response = agent(prompt)
    print()

    # Step 4: Save this turn so future queries can recall it
    response_text = str(response) if not isinstance(response, str) else response
    save_to_memory(user_input, response_text)
