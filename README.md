# ✍️ Essay Generation Pipeline with LangGraph + LangChain + Tavily

This project is an intelligent essay-writing assistant built using LangGraph, LangChain, and the Tavily Search API. It plans, researches, drafts, critiques, and revises essays automatically based on a user-provided topic.


## 📦 Features

- 📑 Generates a high-level essay outline from a user topic.
- 🔍 Performs web research using Tavily API.
- 📝 Writes 5-paragraph essays using LangChain and OpenAI.
- 🧠 Reflects and critiques essays to improve quality.
- 🔁 Supports multi-round revisions (configurable).

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/essay-writer-using-langgraph.git
cd essay-writer-using-langgraph
```

### 2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Add Environment Variables
Create a .env file in the root directory and add your Tavily API key:
TAVILY_API_KEY=your_api_key_here


## 🛠 Tech Stack
- LangGraph - Graph-based orchestration for LLM agents

- LangChain - LLM framework for prompt management and structured output

- Tavily API - Real-time web search for relevant essay content

- OpenAI GPT-3.5 - Essay generation and critique

## 🧪 How to Run
python essay_writer_using_langgraph.py

##📎 Example Task
{"task": "What is the difference between LangChain and LangGraph?", "revision_number": 1, "max_revision": 2}

## 🙌 Acknowledgements
- LangGraph
- LangChain
- Tavily

