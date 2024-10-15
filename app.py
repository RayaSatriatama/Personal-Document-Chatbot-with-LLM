import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
import os
import tempfile
import logging

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cached function for loading embeddings model
@st.cache_resource
def load_embeddings(model_name):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings model {model_name}: {e}")
        st.error("Failed to load embeddings model. Please try again.")
        return None

# Cached function for loading language models
@st.cache_resource
def load_llm(repo_id, temperature, max_length):
    try:
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": temperature, "max_length": max_length}
        )
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM {repo_id}: {e}")
        st.error("Failed to load language model. Please try again.")
        return None

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            logger.error(f"Error reading {pdf.name}: {e}")
            st.warning(f"Could not read {pdf.name}. Skipping.")
    return text

# Split the extracted text into manageable chunks
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store based on the text chunks and embeddings
def get_vector_store(text_chunks, embeddings):
    try:
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        st.error("Failed to create vector store. Please try again.")
        return None

# Initialize the conversation chain with the vector store and language model
def get_conversation_chain(vector_store, llm):
    try:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vector_store.as_retriever(), memory=memory)
        return conversation_chain
    except Exception as e:
        logger.error(f"Error initializing conversation chain: {e}")
        st.error("Failed to initialize conversation chain. Please try again.")
        return None

# Handle user's question and generate a response
def handle_user_question(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process your documents before asking a question.")
    else:
        try:
            with st.spinner("Generating response..."):
                response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history = response.get("chat_history", [])

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error handling user question: {e}")
            st.error("An error occurred while processing your question. Please try again.")

# Clear the chat history
def clear_chat():
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

# Main function to define Streamlit interface
def main():
    load_dotenv()
    st.set_page_config(page_title="Personal Document Chatbot", page_icon=":robot:", layout="wide")

    st.markdown(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Personal Document Chatbot 🤖")

    # Display uploaded documents
    if "uploaded_files" in st.session_state:
        st.subheader("Uploaded Documents")
        for uploaded_file in st.session_state.uploaded_files:
            st.write(f"- {uploaded_file.name}")
    else:
        st.session_state.uploaded_files = []

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_user_question(user_question)
        st.button("Clear Chat", on_click=clear_chat)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type=["pdf"])
        st.markdown("""
            <style>
            .stButton > button {
                width: 100%;
            }
            </style>
            """, unsafe_allow_html=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Display uploaded document names
                    st.session_state.uploaded_files = pdf_docs

                    # Save uploaded files to a temporary directory for security
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        saved_files = []
                        for pdf in pdf_docs:
                            file_path = os.path.join(tmpdirname, pdf.name)
                            try:
                                with open(file_path, "wb") as f:
                                    f.write(pdf.getbuffer())
                                saved_files.append(file_path)
                            except Exception as e:
                                logger.error(f"Error saving {pdf.name}: {e}")
                                st.warning(f"Could not save {pdf.name}. Skipping.")

                        # Extract text
                        raw_text = get_pdf_text(pdf_docs)

                        if not raw_text:
                            st.error("No text could be extracted from the uploaded documents.")
                            return

                        # Split text into chunks
                        text_chunks = get_text_chunks(raw_text)

                        if not text_chunks:
                            st.error("Failed to split text into chunks.")
                            return

                        # Load embeddings
                        embeddings = load_embeddings(model_name="all-MiniLM-L6-v2")  # Using smaller model
                        if embeddings is None:
                            return

                        # Create vector store
                        vector_store = get_vector_store(text_chunks, embeddings)
                        if vector_store is None:
                            return

                        # Load LLM
                        llm = load_llm(repo_id="meta-llama/Llama-3.2-3B", temperature=0.7, max_length=512)
                        if llm is None:
                            return

                        # Initialize conversation chain
                        conversation_chain = get_conversation_chain(vector_store, llm)
                        if conversation_chain is None:
                            return

                        st.session_state.conversation = conversation_chain
                        st.success("Documents processed and chatbot is ready!")
            else:
                st.error("Please upload PDF documents first.")

        # Customization Options
        st.markdown("-----")
        st.subheader("Customization")
        with st.expander("Advanced Settings"):
            repo_id = st.text_input("Language Model Repo ID", value="meta-llama/Llama-3.2-3B")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            max_length = st.number_input("Max Response Length", min_value=100, max_value=2048, value=512)
            model_name = st.selectbox("Embeddings Model", options=["hkunlp/instructor-xl", "all-MiniLM-L6-v2"])

            if st.button("Apply Settings"):
                if st.session_state.conversation:
                    embeddings = load_embeddings(model_name=model_name)
                    if embeddings is None:
                        st.error("Failed to load embeddings model.")
                        return
                    vector_store = st.session_state.conversation.retriever.vector_store
                    vector_store = FAISS.from_texts(texts=vector_store.texts, embedding=embeddings)  # Recreate vector store
                    llm = load_llm(repo_id=repo_id, temperature=temperature, max_length=max_length)
                    if llm is None:
                        return
                    st.session_state.conversation = get_conversation_chain(vector_store, llm)
                    st.success("Settings applied successfully!")
                else:
                    st.warning("Process documents first before applying settings.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
