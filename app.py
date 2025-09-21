import os
import streamlit as st
import asyncio
import uuid
from dotenv import load_dotenv

# Import all the necessary LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# --- NEW SECURE API KEY AND PATH CONFIGURATION ---
# This line loads the variables from your .env file for local development
load_dotenv()

# Check if running on Streamlit Cloud or locally
IS_DEPLOYED = "STREAMLIT_SERVER_RUNNING" in os.environ or "STREAMLIT_SERVER_IS_RUNNING" in os.environ

if IS_DEPLOYED:
    # Load secrets from Streamlit's secrets manager for deployed app
    print("Running on Streamlit Cloud, loading from st.secrets...")
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
    PDF_PATH = st.secrets.get("PDF_PATH", "pdfs") # Default to "pdfs" if not set
else:
    # Load secrets from the .env file for local development
    print("Running locally, loading from .env file...")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    PDF_PATH = os.getenv("PDF_PATH")

# Now, set the environment variables for LangChain to use
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
# --- END OF NEW CONFIGURATION ---


# --- HELPER FUNCTION ---
def is_unhelpful(answer: str) -> bool:
    """Simple check to see if an answer is a refusal or unhelpful."""
    unhelpful_phrases = [
        "i cannot answer", "i can't answer", "i do not have enough information",
        "i cannot find enough information", "i am sorry, i cannot", "the context does not provide"
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in unhelpful_phrases)


# --- CORE CHATBOT LOGIC ---
@st.cache_resource
def setup_search_tool():
    """
    Sets up the entire RAG pipeline and returns the unified search tool.
    This is cached so it only runs once per session.
    """
    print("\n--- Setting up the search tool ---")
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not GOOGLE_API_KEY or not TAVILY_API_KEY:
        st.error("API Key Error: Please set your GOOGLE_API_KEY and TAVILY_API_KEY.")
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    FAISS_INDEX_PATH = "faiss_index"
    vector_store = None

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("-> FAISS index loaded successfully.")
        st.sidebar.success("✅ Knowledge base loaded from disk.")
    else:
        if not os.path.exists(PDF_PATH):
            os.makedirs(PDF_PATH)
            st.info(f"Created '{PDF_PATH}' directory. Please upload your files and refresh the app.")
            return None

        with st.spinner("Processing PDFs for the first time... This may take a while."):
            print("Step 1: Loading PDF documents...")
            # IMPORTANT CHANGE: Using the PDF_PATH variable here
            loader = PyPDFDirectoryLoader(PDF_PATH)
            docs = loader.load()
            print(f"-> Loaded {len(docs)} pages from PDFs.")
            if docs:
                print("Step 2: Splitting documents into smaller chunks...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)
                split_docs = text_splitter.split_documents(docs)
                print(f"-> Split documents into {len(split_docs)} chunks.")

                print("Step 3: Enriching chunks with additional metadata...")
                for i, doc in enumerate(split_docs):
                    doc.metadata["chunk_id"] = f"chunk_{i}"
                print("-> Metadata enrichment complete.")

                print("Step 4: Creating and saving the FAISS vector store...")
                vector_store = FAISS.from_documents(split_docs, embeddings)
                vector_store.save_local(FAISS_INDEX_PATH)
                print(f"-> Vector store created and saved to {FAISS_INDEX_PATH}.")
                st.sidebar.success(f"✅ Indexed {len(docs)} PDF pages and saved to disk.")

    if not vector_store:
        st.sidebar.warning("No knowledge base found. Please add PDFs and restart.")
        return None

    def unified_search(query: str) -> str:
        """
        A comprehensive function that first searches internal documents and, if no answer is found,
        then searches the web. It formats the final answer based on the source.
        """
        print(f"--- Running Unified Search for query: {query} ---")

        print("-> Searching internal documents...")
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                             base_retriever=multi_query_retriever)
        retrieved_docs = compression_retriever.invoke(query)

        if retrieved_docs:
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            summary_prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Based on the context from internal documents provided below, you must provide an answer in two distinct parts.\n\n"
                "#### SUMMARY ####\n"
                "First, provide a brief, one-paragraph summary of the answer.\n\n"
                "#### DETAILS ####\n"
                "Second, provide a detailed, point-by-point breakdown of all the key information that answers the user's question. Use bullet points for clarity.\n\n"
                "If the answer is not in the context, you must say 'I am sorry, I cannot find enough information in the documents to answer this question.'\n\n"
                "--- CONTEXT ---\n"
                "{context}\n\n"
                "--- USER QUESTION ---\n"
                "{question}"
            )
            summary_chain = summary_prompt | llm
            summarized_answer = summary_chain.invoke({"question": query, "context": context}).content

            if not is_unhelpful(summarized_answer):
                sources = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
                unique_sources = sorted(list(set(sources)))
                sources_markdown = "\n\n**Sources from internal repository:**\n" + "\n".join(
                    f"- {os.path.basename(source)}" for source in unique_sources)
                final_answer = "I have searched through the internal repository and found the following:\n\n" + summarized_answer + sources_markdown
                print("-> Found a relevant answer in the documents.")
                return final_answer

        print("-> No definitive answer in documents. Searching the web...")
        raw_search_tool = TavilySearch(max_results=3)
        search_results = raw_search_tool.invoke({"query": query})

        if not search_results:
            return "I could not find a relevant answer in your documents or on the web."

        web_context = "\n\n".join([res.get("content", "") for res in search_results if isinstance(res, dict)])
        web_summary_prompt = ChatPromptTemplate.from_template(
            "Based on the following search results, provide a complete and comprehensive answer to the user's question. Ensure the answer is not cut off.\n\n"
            "User Question: {question}\n\n"
            "Search Results:\n---\n{context}\n---"
        )
        web_summary_chain = web_summary_prompt | llm
        web_summarized_answer = web_summary_chain.invoke({"question": query, "context": web_context}).content

        final_answer = "I could not find the same through the internal data repository, so the result you will be seeing is from the internet:\n\n" + web_summarized_answer
        return final_answer

    print("--- Search tool setup complete! ---")
    return unified_search


# --- STREAMLIT UI ---
st.title("📄 IntelliKMS")
st.info("Ask questions about your documents or get answers from the web.")

unified_search_function = setup_search_tool()

if unified_search_function:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_content = unified_search_function(prompt)
                except Exception as e:
                    response_content = f"An error occurred: {e}"
                st.markdown(response_content)

        st.session_state.messages.append({"role": "assistant", "content": response_content})
else:

    st.warning("Chatbot could not be initialized. Please check the setup instructions and API keys.")
