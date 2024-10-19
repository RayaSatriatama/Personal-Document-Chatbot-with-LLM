import streamlit as st
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
import re
import html

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state variables to prevent AttributeError
st.session_state.setdefault("conversation", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("text_chunks", [])
st.session_state.setdefault("is_generating_response", False)
st.session_state.setdefault("cancel_generation", False)
st.session_state.setdefault("hf_api_token", "")
st.session_state.setdefault("pdf_docs", [])

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

def render_chat_history(chat_history_placeholder):
    """
    Render the chat history in the provided placeholder.
    """
    with chat_history_placeholder.container():
        st.subheader("Chat History")
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.write(user_template.replace("{{MSG}}", chat["content"]), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", chat["content"]), unsafe_allow_html=True)
        else:
            st.write("No chat history yet.")

def render_question_input(question_input_placeholder):
    """
    Render the question input placeholder.
    """
    with question_input_placeholder.container():
        st.subheader("Ask a Question")
        if not st.session_state.is_generating_response:
            with st.form(key='question_form'):
                user_question = st.text_input("Enter your question about the documents:")
                submit_button = st.form_submit_button(label='Submit')
                if submit_button and user_question:
                    handle_user_question(user_question)

def get_pdf_text(pdf_docs):
    """
    Extract text from uploaded PDF documents.
    """
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

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split the extracted text into manageable chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, embeddings):
    """
    Create a vector store based on the text chunks and embeddings.
    """
    try:
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        st.error("Failed to create vector store. Please try again.")
        return None

def get_conversation_chain(vector_store, llm):
    """
    Initialize the conversation chain with the vector store and language model.
    """
    try:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        logger.error(f"Error initializing conversation chain: {e}")
        st.error("Failed to initialize conversation chain. Please try again.")
        return None

def handle_user_question(user_question):
    """
    Handle the user's question and generate a response synchronously.
    """
    if st.session_state.conversation is None:
        st.error("Please upload and process your documents before asking a question.")
        return

    # Add user's question to the chat history immediately
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Start generating response
    st.session_state.is_generating_response = True
    st.session_state.cancel_generation = False

    render_sidebar(st.session_state.is_generating_response)

    try:
        with st.spinner("Generating response..."):
            # Generate response
            response = st.session_state.conversation({"question": user_question})
            logger.info(f"Raw response: {response}")

        # Extract the answer from the response
        raw_chat_history = response.get("chat_history", [])
        if raw_chat_history:
            # Assuming the last message is from the bot
            last_message = raw_chat_history[-1].content.strip()

            # Extract the answer using regex
            match = re.search(r"Answer:\s*(.*)", last_message, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).strip()
            else:
                extracted_answer = last_message

            # Clean any unwanted trailing characters (e.g., non-ASCII characters)
            extracted_answer = re.sub(r'[^\x00-\x7F]+', '', extracted_answer)

            # Append the bot's response to chat history
            st.session_state.chat_history.append({"role": "bot", "content": extracted_answer})
        else:
            extracted_answer = "No response from the model."
            st.session_state.chat_history.append({"role": "bot", "content": extracted_answer})

    except Exception as e:
        logger.error(f"Error handling user question: {e}")
        st.error("An error occurred while processing your question. Please try again.")
    finally:
        # Reset the generating flag
        st.session_state.is_generating_response = False

def handle_cancel():
    """
    Handle cancellation request. Note: Actual cancellation isn't feasible in synchronous code.
    """
    if st.session_state.is_generating_response:
        st.session_state.cancel_generation = True
        st.warning("Cancellation requested. The response will be ignored if it's still processing.")
    else:
        st.warning("No response generation in progress to cancel.")

def clear_chat():
    """
    Clear the chat history and reset relevant session state variables.
    """
    st.session_state.chat_history = []
    st.session_state.conversation = None
    st.session_state.text_chunks = []
    st.session_state.is_generating_response = False
    st.session_state.cancel_generation = False
    st.success("Chat history cleared.")

def render_sidebar(is_generating):
    """
    Render the sidebar content. Uses st.empty for dynamic updates.
    """
    sidebar_placeholder = st.sidebar.empty()

    with sidebar_placeholder.container():
        st.sidebar.subheader("Configuration")

        if not is_generating:
            # Secure input for HuggingFace API Token
            hf_api_token = st.sidebar.text_input(
                "HuggingFace API Token",
                type="password",
                help="Enter your HuggingFace API token to access models.",
                key="hf_api_token_input"
            )
            if hf_api_token:
                st.session_state.hf_api_token = hf_api_token
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token
            else:
                st.sidebar.warning("Please enter your HuggingFace API token to proceed.")

            st.sidebar.markdown("-----")

            # Document Uploader
            pdf_docs = st.sidebar.file_uploader(
                "Upload your PDFs here",
                accept_multiple_files=True,
                type=["pdf"],
                key="pdf_uploader"
            )
            st.session_state.pdf_docs = pdf_docs

            # Display uploaded documents
            st.sidebar.subheader("Your Documents")
            if st.session_state.pdf_docs:
                for uploaded_file in st.session_state.pdf_docs:
                    st.sidebar.write(f"- {uploaded_file.name}")

                # Add "Process Documents" Button
                if st.sidebar.button("Process Documents"):
                    main_process_documents()
            else:
                st.sidebar.write("No documents uploaded.")
                st.sidebar.warning("Please upload PDF documents to process.")

            st.sidebar.markdown("-----")

            # Customization Options
            st.sidebar.subheader("Customization")
            with st.sidebar.expander("Advanced Settings"):
                repo_id = st.sidebar.text_input(
                    "Language Model Repo ID",
                    value="meta-llama/Llama-3.2-3B",
                    key="repo_id_input"
                )
                temperature = st.sidebar.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="temperature_slider"
                )
                max_length = st.sidebar.number_input(
                    "Max Response Length",
                    min_value=100,
                    max_value=2048,
                    value=1024,
                    step=100,
                    key="max_length_input"
                )
                model_name = st.sidebar.selectbox(
                    "Embeddings Model",
                    options=["hkunlp/instructor-xl", "all-MiniLM-L6-v2"],
                    key="model_name_select"
                )
                st.session_state.customization = {
                    "repo_id": repo_id,
                    "temperature": temperature,
                    "max_length": max_length,
                    "model_name": model_name
                }
        else:
            # Sidebar remains visible but inputs are locked (displayed as static text)
            st.sidebar.subheader("Configuration (Locked)")

            # Display current HuggingFace API Token as masked
            if st.session_state.hf_api_token:
                masked_token = "•" * min(len(st.session_state.hf_api_token), 20)
                st.sidebar.text("HuggingFace API Token:")
                st.sidebar.text(masked_token)
            else:
                st.sidebar.text("HuggingFace API Token: Not Set")

            st.sidebar.markdown("-----")

            # Display uploaded documents
            st.sidebar.subheader("Your Documents")
            if "pdf_docs" in st.session_state and st.session_state.pdf_docs:
                for uploaded_file in st.session_state.pdf_docs:
                    st.sidebar.write(f"- {uploaded_file.name}")
            else:
                st.sidebar.write("No documents uploaded.")

            st.sidebar.markdown("-----")

            # Customization Options Displayed as Static Text
            st.sidebar.subheader("Customization")
            with st.sidebar.expander("Advanced Settings"):
                customization = st.session_state.get("customization", {})
                st.sidebar.text(f"Language Model Repo ID: {customization.get('repo_id', 'Not Set')}")
                st.sidebar.text(f"Temperature: {customization.get('temperature', 'Not Set')}")
                st.sidebar.text(f"Max Response Length: {customization.get('max_length', 'Not Set')}")
                st.sidebar.text(f"Embeddings Model: {customization.get('model_name', 'Not Set')}")

def main_process_documents():
    """
    Function to process uploaded documents (called from the sidebar).
    """
    hf_api_token = st.session_state.get("hf_api_token", "")
    if not hf_api_token:
        st.error("HuggingFace API token is required to process documents.")
        return

    pdf_docs = st.session_state.get("pdf_docs", [])
    if pdf_docs:
        with st.spinner("Processing..."):
            # Save uploaded documents temporarily
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

                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text:
                    st.error("No text could be extracted from the uploaded documents.")
                    return

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)
                st.session_state.text_chunks = text_chunks  # Save text_chunks in session state

                if not text_chunks:
                    st.error("Failed to split text into chunks.")
                    return

                # Load embeddings
                customization = st.session_state.get("customization", {})
                model_name = customization.get("model_name", "all-MiniLM-L6-v2")
                embeddings = load_embeddings(model_name=model_name)  # Use customizable model
                if embeddings is None:
                    return

                # Create vector store
                vector_store = get_vector_store(text_chunks, embeddings)
                if vector_store is None:
                    return

                # Load LLM with customizable settings
                repo_id = customization.get("repo_id", "meta-llama/Llama-3.2-3B")
                temperature = customization.get("temperature", 0.5)
                max_length = customization.get("max_length", 1024)
                llm = load_llm(repo_id=repo_id, temperature=temperature, max_length=max_length)
                if llm is None:
                    return

                # Initialize conversation chain without custom prompt
                conversation_chain = get_conversation_chain(vector_store, llm)
                if conversation_chain is None:
                    return

                st.session_state.conversation = conversation_chain
                st.success("Documents processed and chatbot is ready!")
    else:
        st.error("Please upload PDF documents first.")

def main():
    """
    Main function to define the Streamlit interface.
    """
    st.set_page_config(page_title="Personal Document Chatbot", page_icon=":robot:", layout="wide")

    st.markdown(css, unsafe_allow_html=True)

    st.header("Personal Document Chatbot 🤖")

    # Create placeholders for different sections
    chat_history_placeholder = st.empty()
    question_input_placeholder = st.empty()
    status_placeholder = st.empty()
    clear_chat_placeholder = st.empty()

    # Use the placeholder to display chat history
    render_chat_history(chat_history_placeholder)
    render_question_input(question_input_placeholder)

    # Render the sidebar only once using st.empty
    sidebar_placeholder = st.sidebar.empty()
    with sidebar_placeholder.container():
        render_sidebar(st.session_state.is_generating_response)

    # Clear Chat Button using st.empty
    with clear_chat_placeholder.container():
        if st.button("Clear Chat"):
            clear_chat()

# Run the Streamlit app
if __name__ == "__main__":
    main()
