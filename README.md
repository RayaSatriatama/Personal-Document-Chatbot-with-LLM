# Personal-Document-Chatbot 🤖

Welcome to the **Personal Document Chatbot**! This project provides a chatbot interface where you can upload PDF documents and ask questions about the content. The chatbot uses powerful language models and vector embeddings to retrieve information from your documents, making it a useful tool for processing large amounts of text quickly.

## Features

- 📄 **PDF Document Upload**: Upload multiple PDF files.
- 🔍 **Text Extraction**: Extracts text from uploaded PDFs.
- 🧠 **Conversational Interface**: Ask questions about the content of your PDFs and get AI-generated responses.
- 🗂️ **Vector Store for Text Retrieval**: Uses FAISS to create vector embeddings of the text chunks for efficient retrieval.
- 🔄 **Session Memory**: The chatbot retains the chat history during the session, allowing for context-based interactions.
- ⚙️ **Customizable Settings**: Modify the language model, response length, and temperature according to your preferences.

## Disclaimer

⚠️ **Hugging Face API Limitations**:
This project uses models from Hugging Face, which may be subject to API usage limits depending on your account type. Free-tier users of Hugging Face may experience restrictions in terms of request limits, rate limits, or model availability. If you encounter errors related to API requests, consider upgrading your Hugging Face account or monitoring your API usage closely.

## Installation

Follow these steps to set up the project locally:

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/) (Python package installer)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Personal-Document-Chatbot.git
   cd Personal-Document-Chatbot
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   - Create a `.env` file in the root directory and add your Hugging Face API key (if required):

     ```bash
     HUGGINGFACEHUB_API_TOKEN=your_token_here
     ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

   Open your browser and go to `http://localhost:8501` to see the chatbot interface.

## Usage

1. **Upload PDFs**: Use the file uploader on the sidebar to upload one or multiple PDF documents.
2. **Process Documents**: Once uploaded, click the "Process" button to extract and process the text.
3. **Ask Questions**: Type your questions in the input field, and the chatbot will respond based on the content of the uploaded PDFs.
4. **Clear Chat**: Use the "Clear Chat" button to reset the conversation.

## Customization

You can adjust the behavior of the chatbot by changing the following settings in the sidebar:

- **Language Model Repo ID**: Select or input a different language model from Hugging Face.
- **Temperature**: Control the randomness of the AI’s responses (higher values give more creative responses).
- **Max Response Length**: Set the maximum length of the chatbot’s responses.
- **Embeddings Model**: Choose between different embedding models for vector store creation.

## Project Structure

```bash
.
├── app.py               # Main application file
├── htmlTemplates.py      # HTML templates for the chatbot interface
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── README.md             # Project documentation
```

## Built With

- [Streamlit](https://streamlit.io/) - Web app framework for the UI
- [Langchain](https://github.com/hwchase17/langchain) - Framework for building language model applications
- [PyPDF2](https://pypdf2.readthedocs.io/en/stable/) - Library for extracting text from PDFs
- [FAISS](https://github.com/facebookresearch/faiss) - Library for efficient similarity search and clustering of dense vectors

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

## Contact

For any questions or inquiries, you can reach me at [your-email@example.com].

---

Enjoy chatting with your documents! 🚀
