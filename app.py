## RAG Q&A Conversation With PDF Including Chat History

import streamlit as st
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

import os
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

import inspect
import streamlit_authenticator as stauth

print(inspect.signature(stauth.Authenticate))
print(inspect.signature(stauth.Authenticate.login))
print(inspect.signature(stauth.Authenticate.logout))

# st.write(stauth.__version__)
# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ==========================================================
# Authentication
# ==========================================================

with open("auth.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

name, authentication_status, username = authenticator.login(
    location="main",
    key="Login"
)
st.write(authentication_status)
st.write(name)
st.write(username)
# ==========================================================
# Embedding Model
# ==========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ==========================================================
# Authentication Status
# ==========================================================

if authentication_status is False:

    st.error("❌ Username or Password is incorrect.")

elif authentication_status is None:

    st.warning("🔐 Please login to continue.")

elif authentication_status:

    # Sidebar
    st.sidebar.success(f"Welcome, {name}")
    st.sidebar.write(f"👤 User : {username}")

    authenticator.logout(
    button_name="Logout",
    location="sidebar"
    )

    # ==========================================================
    # UI
    # ==========================================================

    st.title("📄 Conversational RAG with PDF")

    st.write(
        "Upload one or more PDFs and chat with them."
    )

    api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
    )

    if api_key:

        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0,
        )

        session_id = st.text_input(
            "Session ID",
            value="default_session",
        )

        if "store" not in st.session_state:
            st.session_state.store = {}

        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type="pdf",
            accept_multiple_files=True,
        )

        if uploaded_files:

            documents = []

            for uploaded_file in uploaded_files:

                temp_pdf = "./temp.pdf"

                with open(temp_pdf, "wb") as file:
                    file.write(uploaded_file.getvalue())

                loader = PyPDFLoader(temp_pdf)

                docs = loader.load()

                documents.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,
                chunk_overlap=500,
            )

            splits = text_splitter.split_documents(
                documents
            )

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
            )

            retriever = vectorstore.as_retriever()

            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question, "
                "formulate a standalone question that can be understood "
                "without the chat history. "
                "Do NOT answer the question."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        contextualize_q_system_prompt,
                    ),
                    MessagesPlaceholder(
                        "chat_history"
                    ),
                    (
                        "human",
                        "{input}",
                    ),
                ]
            )

            history_aware_retriever = (
                create_history_aware_retriever(
                    llm,
                    retriever,
                    contextualize_q_prompt,
                )
            )
                        # ==========================================================
            # QA Prompt
            # ==========================================================

            system_prompt = (
                "You are an AI assistant for question answering "
                "using uploaded PDFs.\n\n"

                "Use ONLY the provided context to answer.\n\n"

                "If the user asks to summarize, generate a concise "
                "summary of the complete document.\n\n"

                "If the answer is not available in the context, "
                "reply with 'I don't know.'\n\n"

                "{context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_prompt,
                    ),
                    MessagesPlaceholder(
                        "chat_history"
                    ),
                    (
                        "human",
                        "{input}",
                    ),
                ]
            )

            # ==========================================================
            # Question Answer Chain
            # ==========================================================

            question_answer_chain = (
                create_stuff_documents_chain(
                    llm,
                    qa_prompt,
                )
            )

            # ==========================================================
            # RAG Chain
            # ==========================================================

            rag_chain = create_retrieval_chain(
                history_aware_retriever,
                question_answer_chain,
            )

            # ==========================================================
            # Chat History
            # ==========================================================

            def get_session_history(
                session: str,
            ) -> BaseChatMessageHistory:

                if (
                    session
                    not in st.session_state.store
                ):
                    st.session_state.store[
                        session
                    ] = ChatMessageHistory()

                return st.session_state.store[
                    session
                ]

            conversational_rag_chain = (
                RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
            )

            # ==========================================================
            # User Interface
            # ==========================================================

            col1, col2 = st.columns([5, 1])

            with col1:

                user_input = st.text_input(
                    "Ask a question"
                )

            with col2:

                summarize = st.button(
                    "Summarize"
                )

            if summarize:

                user_input = """
Summarize the complete uploaded PDF.

Include:

1. Overview

2. Main Topics

3. Important Concepts

4. Key Conclusions
"""

            # ==========================================================
            # Invoke RAG
            # ==========================================================

            if user_input:

                session_history = get_session_history(
                    session_id
                )

                response = conversational_rag_chain.invoke(
                    {
                        "input": user_input
                    },
                    config={
                        "configurable": {
                            "session_id": session_id
                        }
                    },
                )

                answer = response["answer"]

                st.subheader(
                    "Assistant Response"
                )

                st.write(answer)

                if summarize:

                    st.success(
                        "Summary Generated Successfully"
                    )

                # =====================================
                # Retrieved Chunks
                # =====================================

                with st.expander(
                    "Retrieved Chunks"
                ):

                    if "context" in response:

                        for i, doc in enumerate(
                            response["context"]
                        ):

                            st.markdown(
                                f"### Chunk {i+1}"
                            )

                            st.write(
                                doc.page_content
                            )

                            st.divider()
                                # =====================================
                # Chat History
                # =====================================

                with st.expander("Chat History"):

                    for msg in session_history.messages:

                        if msg.type == "human":

                            st.markdown(
                                f"**🧑 You:** {msg.content}"
                            )

                        elif msg.type == "ai":

                            st.markdown(
                                f"**🤖 Assistant:** {msg.content}"
                            )

    else:

        st.info(
            "🔑 Please enter your Groq API Key to continue."
        )