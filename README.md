# intellikms-demo
GenAI app to demystify documents

IntelliKMS Demo
IntelliKMS is a Retrieval-Augmented Generation (RAG) application that transforms a collection of PDF documents into an intelligent, conversational knowledge base. Users can ask questions in natural language and receive detailed, sourced answers from their documents, with a fallback to a web search if the information is not found internally.

Features
Internal Knowledge Base: Ingests and indexes multiple PDF documents to create a searchable knowledge base.

Advanced RAG Pipeline: Utilizes a sophisticated retrieval pipeline with Multi-Query Generation and Contextual Compression to find the most relevant information.

Web Search Fallback: If an answer cannot be found in the internal documents, the system automatically performs a web search using Tavily AI to provide a comprehensive response.

Persistent Vector Store: Creates and saves a FAISS vector index, so the time-consuming document processing step only needs to be run once.

Interactive UI: Built with Streamlit to provide a simple and intuitive chat interface.

Tech Stack
Framework: Streamlit

LLM & Embeddings: Google Gemini 1.5 Pro & Google Generative AI Embeddings

Core Logic: LangChain

Vector Store: FAISS (Facebook AI Similarity Search)

Web Search: Tavily AI

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Python 3.8+

Git

1. Clone the Repository
First, clone this repository to your local machine.

git clone [https://github.com/chaitanyachawla97/intellikms-demo.git](https://github.com/chaitanyachawla97/intellikms-demo.git)
cd intellikms-demo

2. Set Up the Knowledge Base
You need to provide the PDF documents that will form the knowledge base.

Create a folder named pdfs in the root of the project directory.

Place all your PDF files inside this pdfs folder.

Your project structure should look like this:

intellikms-demo/
├── app.py
└── pdfs/
    ├── document1.pdf
    └── document2.pdf

3. Install Dependencies
Install all the required Python packages using pip.

pip install streamlit langchain langchain_google_genai langchain_community langchain-text-splitters pypdf faiss-cpu langchain-tavily cryptography

4. Configure API Keys
This application requires API keys from Google and Tavily to function.

Open the app.py file in a text editor.

Locate the following lines near the top of the file:

# --- API KEY CONFIGURATION ---
# IMPORTANT: PASTE YOUR API KEYS DIRECTLY HERE
GOOGLE_API_KEY = "YOUR_GOOGLE_AI_API_KEY_HERE"
TAVILY_API_KEY = "YOUR_TAVILY_API_KEY_HERE"

Replace "YOUR_GOOGLE_AI_API_KEY_HERE" with your actual Google AI API key.

Replace "YOUR_TAVILY_API_KEY_HERE" with your actual Tavily API key.

Save the app.py file.

5. Run the Application
You are now ready to launch the IntelliKMS application.

Open your terminal in the project's root directory.

Run the following command:

streamlit run app.py

Your web browser will automatically open to the application's URL (usually http://localhost:8501).

The first time you run the app, it will process the PDFs and create the faiss_index. This may take a few minutes depending on the number and size of your documents. On subsequent runs, it will load the index from disk, which is much faster.
