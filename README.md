RAG-Based Chatbot with Document Search

This is a web-based conversational AI application that uses Retrieval-Augmented Generation (RAG) for answering questions based on documents (e.g., PDFs) stored in a local directory. The application uses OpenAI's GPT model for generating responses and integrates FAISS for efficient document similarity search. It provides an interactive chat interface and retrieves relevant documents to assist with answering user queries.

Table of Contents

Features
Prerequisites
Setup Instructions
Environment Variables
Running the Application
Folder Structure
Contributing
License
Features

Conversational AI: Responds to user queries using context from uploaded documents.
Document-based Question Answering: Uses FAISS to search for relevant documents and generates responses based on those documents.
History Tracking: Maintains conversation history for personalized interactions.
Source Tracking: Displays the source of each document used in generating answers (i.e., the PDF filename).
Interactive Web Interface: A simple web-based interface powered by Flask to interact with the AI model.
Prerequisites

Before setting up the application, make sure you have the following installed:

Conda (for managing environments)
Python 3.7+
pip (Python package installer)
Flask (for the web interface)
OpenAI API Key (for interacting with OpenAI's GPT model)
FAISS (for vector store and document similarity search)
LangChain (for document processing and question answering)
Setup Instructions

1. Clone the Repository
Clone this repository to your local machine using the following command:

git clone https://github.com/Sharafunneesa-pp/ragchatbot-assist.git
cd ragchatbot-assist
2. Create and Activate Conda Environment
Create a new Conda environment with Python 3.11:

conda create ragchat python==3.11
Activate the environment:

conda activate ragchat
3. Install Dependencies
Once the Conda environment is activated, install the required Python dependencies using pip:

pip install -r requirements.txt
The requirements.txt file contains the following libraries:

Flask
LangChain
FAISS
openai
dotenv
logging
PyPDF2
Any other required libraries listed in your requirements.txt
4. Set Up Environment Variables
Create a .env file in the root directory of your project to store your environment variables. The .env file should include the following:

OPENAI_API_KEY=your_openai_api_key
Replace your_openai_api_key with your actual OpenAI API key. You can get an API key by signing up at OpenAI's platform.

5. Add PDF Files to Data Folder
Create a folder named data in the root of your project. Place any PDF documents you want the bot to use for answering questions inside this folder.

Example folder structure:

ragchatbot-assist/
├── data/
│   ├── document1.pdf
│   └── document2.pdf
├── templates/
│   └── ragchat.html
├── ragchat.py               # RAG implementation with LLM
├── ragchat.ipynb            # Jupyter notebook for RAG implementation
├── app.py                   # Main Flask application file
├── .env                     # Environment variables for API key
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
6. Run the Application
Once the setup is complete and the environment variables are configured, you can run the Flask application by executing the following command:

python app.py
This will start the Flask development server, and the application will be available at http://127.0.0.1:5000/ in your web browser.

Environment Variables

OPENAI_API_KEY: Your OpenAI API key used for GPT model interaction. This is required for generating responses from the AI model.
The .env file should contain the API key like this:

OPENAI_API_KEY=your_openai_api_key
Running the Application

After starting the application, open a web browser and go to http://127.0.0.1:5000/.
Enter your query in the chat interface, and the AI model will provide an answer based on the available documents in the data folder.
The response will be generated using context from relevant documents found in the vector store, and the source (PDF file) will be shown.
Example Interaction
User: "What is Cross-Attention?"

AI: "Cross-attention is a mechanism in transformer models that allows information from one sequence to attend to another sequence. This is especially useful in models like BERT and GPT for tasks like machine translation and text generation, where input sequences can be quite different but related."

Each response is based on the documents you have uploaded to the data folder.

Folder Structure

ragchatbot-assist/
├── data/                # Folder to store PDF documents
│   ├── document1.pdf    # Example document
│   └── document2.pdf    # Another example document
├── templates/           # Folder for HTML templates
│   └── ragchat.html     # HTML template for the chat interface
├── ragchat.py           # RAG implementation with LLM
├── ragchat.ipynb        # Jupyter notebook for RAG implementation
├── app.py               # Main application file
├── .env                 # Environment variables for API key
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
Contributing

We welcome contributions! If you would like to contribute to this project, please fork the repository, make your changes, and submit a pull request. Ensure that all code adheres to the existing style and that any new features are well-documented.
