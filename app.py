import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader 


load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_retriever(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()

        if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={"source": uploaded_file.name, "page": page_number},
                        )
                    )
        else:
            text = file_bytes.decode("utf-8", errors="ignore")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name},
                    )
                )

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def uploaded_files_signature(uploaded_files):
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def create_documents_tool(retriever):
    @tool("uploaded_documents")
    def uploaded_documents(query: str) -> str:
        """Search the uploaded documents for information relevant to the user's question."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant uploaded document content was found."

        chunks = []
        for doc in docs:
            source = doc.metadata.get("source", "uploaded document")
            page = doc.metadata.get("page")
            label = f"{source}, page {page}" if page else source
            chunks.append(f"Source: {label}\n{doc.page_content}")

        return "\n\n".join(chunks)

    return uploaded_documents


arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")


def default_messages():
    return [
        {
            "role": "assistant",
            "content": "Hi, choose tools from the sidebar and ask me anything.",
        }
    ]


st.set_page_config(page_title="LangChain Enhanced Tools Chat", page_icon="🔎")
st.title("🔎 LangChain Chat with Selectable Tools")
st.write(
    "Choose the tools you want to enable, then ask questions in the chat. "
    "When document chat is enabled, upload files in the sidebar first."
)

with st.sidebar:
    st.header("Settings")
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("Groq API key loaded from .env.")
    else:
        st.warning("GROQ_API_KEY is missing from .env or the environment.")

    st.header("Tools")
    use_search = st.checkbox("Search", value=True)
    use_wiki = st.checkbox("Wikipedia", value=True)
    use_arxiv = st.checkbox("Arxiv", value=True)
    use_documents = st.checkbox("Uploaded documents", value=False)

    uploaded_files = []
    if use_documents:
        uploaded_files = st.file_uploader(
            "Add document/s",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload PDFs or text files to chat against them.",
        )


current_tool_selection = {
    "search": use_search,
    "wiki": use_wiki,
    "arxiv": use_arxiv,
    "documents": use_documents,
}

if "tool_selection" not in st.session_state:
    st.session_state["tool_selection"] = current_tool_selection
elif st.session_state["tool_selection"] != current_tool_selection:
    st.session_state["tool_selection"] = current_tool_selection
    st.session_state["messages"] = default_messages()
    st.session_state["chat_memory"] = []
    st.toast("Tool selection changed. Chat was reinitialized.")


if "messages" not in st.session_state:
    st.session_state["messages"] = default_messages()

if "chat_memory" not in st.session_state:
    st.session_state["chat_memory"] = []

if "document_retriever_signature" not in st.session_state:
    st.session_state["document_retriever_signature"] = None

if "document_retriever" not in st.session_state:
    st.session_state["document_retriever"] = None


enabled_tools = []
if use_search:
    enabled_tools.append(search)
if use_wiki:
    enabled_tools.append(wiki)
if use_arxiv:
    enabled_tools.append(arxiv)

if use_documents:
    if uploaded_files:
        signature = uploaded_files_signature(uploaded_files)
        if signature != st.session_state["document_retriever_signature"]:
            with st.sidebar.spinner("Indexing uploaded documents..."):
                st.session_state["document_retriever"] = build_retriever(uploaded_files)
                st.session_state["document_retriever_signature"] = signature

        if st.session_state["document_retriever"]:
            enabled_tools.append(create_documents_tool(st.session_state["document_retriever"]))
            st.sidebar.success("Documents are ready for chat.")
        else:
            st.sidebar.warning("No readable text was found in the uploaded documents.")
    else:
        st.sidebar.info("Upload document/s to enable the document chat tool.")
else:
    st.session_state["document_retriever"] = None
    st.session_state["document_retriever_signature"] = None


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input(placeholder="What is machine learning?"):
    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to your environment or .env file.")
        st.stop()

    if not enabled_tools:
        st.error("Select at least one tool from the sidebar before chatting.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state["chat_memory"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True)
    search_agent = create_agent(
        model=llm,
        tools=enabled_tools,
        system_prompt=(
            "You are a helpful assistant. Use only the enabled tools when they are useful. "
            "If uploaded documents are enabled and the user asks about their files, use the "
            "uploaded_documents tool before answering. Provide concise answers and mention "
            "document sources when using uploaded document content."
        ),
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        result = search_agent.invoke(
            {"messages": st.session_state["chat_memory"]},
            config={"callbacks": [st_cb]},
        )
        response = result["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state["chat_memory"].append({"role": "assistant", "content": response})
        st.write(response)
