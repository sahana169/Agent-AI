# Challenge 1 - Simple AI Chatbot using Strands SDK + Ollama (llama3.2:3b)
# Strands Agents SDK: https://strandsagents.com

from strands import Agent
from strands.models.ollama import OllamaModel

# Create the Ollama model instance
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

# Create the agent with a system prompt
# Strands automatically keeps conversation history across turns
agent = Agent(
    model=ollama_model,
    system_prompt="You are a helpful assistant. Keep your answers clear and beginner-friendly.",
)

print("Chatbot ready! Type 'exit' or 'quit' to stop.\n")

# Chat loop — keeps running until the user types exit/quit
while True:
    user_input = input("You: ").strip()

    # Skip empty input
    if not user_input:
        continue

    # Exit condition
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # Send the message to the agent — it remembers previous turns automatically
    print("Assistant: ", end="", flush=True)
    agent(user_input)
    print()  # Newline after each response
