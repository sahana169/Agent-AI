# AI Agents with Strands SDK, Ollama, Mem0, FAISS, and MCP

## Overview

This repository contains my solutions for the AWS Builders Skill Sprint AI Agent Challenges. The projects demonstrate how to build increasingly capable AI agents using Strands SDK and Ollama (llama3.2:3b), starting from a simple chatbot and progressing to agents with tools, memory, and MCP integration.

## Technologies Used

- Python
- Strands SDK
- Ollama
- Llama 3.2 (3B)
- Mem0
- FAISS
- MCP (Model Context Protocol)

---

## Challenges Completed

### Challenge 1: Basic AI Agent

Built a simple conversational AI agent using:

- Strands SDK
- Ollama
- llama3.2:3b

**Features**
- Interactive chat interface
- Local LLM execution
- Continuous conversation loop

---

### Challenge 2: Tools Agent

Enhanced the agent with custom tools.

**Tools Added**
- Calculator Tool
- Weather Tool
- Age Calculator Tool

**Features**
- Tool invocation through the agent
- Multi-turn conversations
- Interactive command-line chatbot

---

### Challenge 3: Memory Agent

Added persistent memory capabilities using Mem0 and FAISS.

**Features**
- Store user information
- Retrieve stored memories
- Persistent conversations
- User information recall

**Example**

**User:** My name is Thamarai

**Agent:** Nice to meet you, Thamarai.

**Later...**

**User:** What is my name?

**Agent:** Your name is Thamarai.

---

### Challenge 4: Full Agent

Combined all previous functionalities into a single agent.

**Features**
- Calculator Tool
- Weather Tool
- Age Calculator Tool
- Persistent Memory
- User Information Recall
- Interactive Chat Loop

---

### Challenge 5: MCP Chatbot

Built an MCP-enabled chatbot using Strands SDK.

**Features**
- MCP Server Connection
- MCP Tool Usage
- Interactive Conversations
- Extensible Agent Architecture

---

## Project Structure

```text
Challenge-1/
│── starter.py

Challenge-2/
│── starter.py

Challenge-3/
│── starter.py

Challenge-4/
│── starter.py

Challenge-5/
│── starter.py
```

---

## Installation

### Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### Install Dependencies

```bash
pip install strands-agents
pip install ollama
pip install "mem0ai[nlp]"
pip install faiss-cpu
```

### Pull the Ollama Model

```bash
ollama pull llama3.2:3b
```

---

## Running the Projects

Navigate to the required challenge folder and run:

```bash
cd Challenge-1
python starter.py
```

Repeat for other challenges:

```bash
cd Challenge-2
python starter.py
```

```bash
cd Challenge-3
python starter.py
```

```bash
cd Challenge-4
python starter.py
```

```bash
cd Challenge-5
python starter.py
```

---

## Key Learnings

Through these challenges, I gained hands-on experience with:

- AI Agent Development
- Local LLM Deployment
- Tool Calling
- Persistent Memory Systems
- Vector Databases (FAISS)
- MCP Integration
- Agentic AI Workflows

---

## Future Improvements

- Integrate AWS services such as Amazon Bedrock
- Add Retrieval-Augmented Generation (RAG)
- Support multiple LLM providers
- Build a web-based UI
- Deploy agents on the cloud

---

## Acknowledgements

Special thanks to the AWS Builders Skill Sprint program for providing practical, hands-on challenges that helped strengthen my understanding of modern AI agent development.

---

## Author

**Sahana Srinivasan**

Cloud Computing Enthusiast | AI Agent Developer | AWS Builder
