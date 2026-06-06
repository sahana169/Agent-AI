# Challenge 5 - MCP Chatbot using Strands SDK + Ollama + Local MCP Server
#
# Architecture:
#   starter.py  ──►  Strands Agent (llama3.2:3b via Ollama)
#                         └──►  MCPClient  ──►  mcp_server.py (subprocess)
#                                                  Tools: calculator, get_datetime,
#                                                         calculate_age, unit_converter
#
# MCP (Model Context Protocol) is an open standard for connecting AI agents to
# external tool servers. The server runs as a subprocess; Strands manages it.
# They communicate via JSON messages over stdin/stdout (stdio transport).
#
# ─── SETUP (run once) ─────────────────────────────────────────────────────────
#   pip install "strands-agents[ollama]" mcp
#   ollama pull llama3.2:3b
# ──────────────────────────────────────────────────────────────────────────────

import sys
import os
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools.mcp import MCPClient
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

# =============================================================================
# SECTION 1: MCP CLIENT SETUP
#
# StdioServerParameters tells MCPClient how to launch the MCP server process:
#   command  → Python executable (same one running this file)
#   args     → [path to mcp_server.py]
#
# MCPClient wraps the connection and handles:
#   - spawning mcp_server.py as a child process
#   - the MCP handshake (initialize / list_tools)
#   - forwarding tool calls from the agent to the server and returning results
#
# The lambda `lambda: stdio_client(server_params)` is a factory — MCPClient
# calls it when it needs to (re)connect to the server.
# =============================================================================
server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")

server_params = StdioServerParameters(
    command=sys.executable,  # e.g. "C:/Python313/python.exe"
    args=[server_path],      # runs: python mcp_server.py
)

mcp_client = MCPClient(lambda: stdio_client(server_params))

# =============================================================================
# SECTION 2: OLLAMA MODEL
#
# llama3.2:3b running locally via Ollama — same as all previous challenges.
# =============================================================================
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

# =============================================================================
# SECTION 3: AGENT + MCP TOOLS
#
# `with mcp_client:` starts the MCP server subprocess and keeps it alive for
# the entire session. It shuts it down cleanly when the block exits.
#
# `mcp_client.list_tools_sync()` queries the server and returns MCPAgentTool
# objects — one per tool registered in mcp_server.py.
#
# Passing these to Agent(tools=[...]) works exactly like @tool functions.
# The agent reads each tool's name, description, and parameter schema
# directly from the MCP server — no manual wiring needed.
# =============================================================================
print("Starting MCP server and loading tools...\n")

with mcp_client:
    # Fetch all tools from the running MCP server
    tools = mcp_client.list_tools_sync()

    agent = Agent(
        model=ollama_model,
        system_prompt=(
            "You are a helpful assistant connected to a local MCP tool server. "
            "Use calculator for math, get_datetime for date/time questions, "
            "calculate_age for age-related questions, and unit_converter for unit conversions. "
            "Always use a tool when the question calls for one."
        ),
        tools=tools,
    )

    # =========================================================================
    # SECTION 4: INTERACTIVE CHAT LOOP
    #
    # Must stay inside `with mcp_client:` so the server process stays alive.
    # The agent automatically decides which MCP tool to call based on the input.
    # =========================================================================
    print(f"MCP Chatbot ready! {len(tools)} tools loaded from MCP server.")
    print("Tools:", ", ".join(t.tool_name for t in tools))
    print("\nTry:")
    print("  'What is sqrt(144) + 50?'")
    print("  'What day is today?'")
    print("  'How old is someone born on March 10, 2000?'")
    print("  'Convert 100 celsius to fahrenheit'")
    print("  'Convert 10 km to miles'")
    print("\nType 'tools' to list MCP tools. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Show available MCP tools on demand
        if user_input.lower() == "tools":
            print("\n--- MCP Tools Available ---")
            for t in tools:
                print(f"  • {t.tool_name}")
            print("---------------------------\n")
            continue

        # Agent handles the message — calls MCP tools automatically when needed
        print("Assistant: ", end="", flush=True)
        agent(user_input)
        print()
