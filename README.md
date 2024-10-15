# Personal-Document-Chatbot 🤖

The **Personal Document Chatbot** allows you to upload PDF documents, process their content, and interact with a chatbot that can answer questions based on the uploaded documents. The chatbot leverages Hugging Face language models and FAISS for vector-based information retrieval.

[Personal Document Chatbot](https://personal-document-chatbot-with-llm.streamlit.app/)

## Features

- 📄 **PDF Document Upload**: Upload multiple PDF documents.
- 🧠 **Conversational Interface**: Ask questions about the documents and receive intelligent responses.
- 🔍 **Text Extraction**: Extracts and processes text from the uploaded PDFs.
- 🔄 **Chat Memory**: Retains conversation history during the session for context-based answers.
- 🧬 **Customizable AI Settings**: Adjust the language model, temperature, response length, and embeddings model to your needs.
- 🗂️ **Efficient Vector Search**: Uses FAISS for fast similarity searches on document content.

## Disclaimer

⚠️ **Hugging Face API Limitations**: This project uses Hugging Face models, which may have usage limits depending on your account type. Free-tier accounts may face API rate limits or request restrictions. Ensure you have a valid Hugging Face API token to use the chatbot.

## Installation

To set up the project locally, follow these steps:

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/) (Python package installer)
- Hugging Face API token (sign up at [Hugging Face](https://huggingface.co/) if you don't have one)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Personal-Document-Chatbot.git
   cd Personal-Document-Chatbot
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   You can either use a `.env` file or directly enter your Hugging Face API token into the application via the sidebar.

   To use a `.env` file, create it in the root directory and add:

   ```bash
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

   Open your browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Usage

1. **Upload PDFs**: Use the file uploader in the sidebar to upload one or more PDF documents.
2. **Process Documents**: Once uploaded, click the "Process" button to extract and process the text.
3. **Ask Questions**: Type your question in the input field and the chatbot will provide answers based on the document content.
4. **Clear Chat**: Use the "Clear Chat" button to reset the conversation.

### Hugging Face API Token

To use the chatbot, you must provide a valid Hugging Face API token. You can enter it directly into the app through the sidebar. Without this token, the chatbot will not function as it relies on models hosted by Hugging Face.

## Customization

You can customize the chatbot's behavior using the following options in the sidebar:

- **Language Model Repo ID**: Choose or input a Hugging Face model ID.
- **Temperature**: Adjust the creativity of the responses (higher values lead to more random responses).
- **Max Response Length**: Set the maximum token length for responses.
- **Embeddings Model**: Select an embeddings model for vector-based retrieval, such as `all-MiniLM-L6-v2` or `hkunlp/instructor-xl`.

## Project Structure

```bash
.
├── app.py               # Main application file
├── htmlTemplates.py      # HTML templates for the chatbot interface
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (optional)
└── README.md             # Project documentation
```

## Built With

- [Streamlit](https://streamlit.io/) - Web framework for creating the UI
- [Langchain](https://github.com/hwchase17/langchain) - Framework for building language model applications
- [PyPDF2](https://pypdf2.readthedocs.io/en/stable/) - PDF text extraction library
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search and clustering library
- [Hugging Face](https://huggingface.co/) - Platform providing language models and embeddings

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create a pull request or open an issue.

### Steps to Contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit the changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Enjoy using the **Personal Document Chatbot** to interact with your documents! 🚀