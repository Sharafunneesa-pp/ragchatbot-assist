import os
import logging
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import openai

# Load environment variables from a .env file
load_dotenv()

# Fetch OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key for usage
openai.api_key = OPENAI_API_KEY

# Initialize the Flask application
app = Flask(__name__)

# Initialize global variables for conversation history and source tracking
conversation_history = []
source_tracking = []

# Import Document class for handling document-related metadata
from langchain.schema import Document

# Setup logging for error handling and debugging purposes
logging.basicConfig(level=logging.ERROR)

# Function to load PDF documents from a given folder and split them into chunks
def get_pdf_text_from_folder(pdf_folder):
    documents = []
    try:
        # Loop through each file in the folder and process only PDF files
        for pdf_file in os.listdir(pdf_folder):
            if pdf_file.endswith(".pdf"):  # Only process PDF files
                pdf_path = os.path.join(pdf_folder, pdf_file)
                
                # Use LangChain's PyPDFLoader to load and split the PDF
                loader = PyPDFLoader(pdf_path)
                pages = loader.load_and_split()

                # For each page, create a Document object containing the text and metadata (source filename)
                for page in pages:
                    chunk = page.page_content
                    document = Document(page_content=chunk, metadata={'source': pdf_file})
                    documents.append(document)
    except Exception as e:
        logging.error(f"Error in get_pdf_text_from_folder: {str(e)}")
    return documents

# Function to split large chunks of text into smaller parts for easier processing
def get_text_chunks(text):
    try:
        # Use LangChain's RecursiveCharacterTextSplitter to split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)
    except Exception as e:
        logging.error(f"Error in get_text_chunks: {str(e)}")
        return []

# Function to create a vector store using FAISS and OpenAI embeddings from the documents
def create_vector_store(pdf_folder):
    try:
        # Load documents from the PDF folder
        documents = get_pdf_text_from_folder(pdf_folder)

        # Use OpenAI embeddings to represent text data as vectors
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        # Use FAISS to create a vector store from the documents
        vector_store = FAISS.from_documents(documents, embedding=embeddings)

        # Save the vector store locally for future use
        vector_store.save_local("faiss_index")

        return vector_store
    except Exception as e:
        logging.error(f"Error in create_vector_store: {str(e)}")
        return None

# Function to create a retrieval chain for question answering using the LLM
def get_conversational_chain():
    try:
        # Define a prompt template to combine context and chat history with the user's question
        prompt_template = """
        Use the provided context and chat history to answer the question accurately. If you can't answer based on the information, please state that.

        Context: {context}
        Chat History: {chat_history}
        Human: {question}
        AI: 
        """
        
        # Initialize the OpenAI model for question answering with specific parameters
        model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=3000)

        # Create a PromptTemplate object with the defined prompt
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])

        # Load the QA chain using the prompt and the OpenAI model
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        logging.error(f"Error in get_conversational_chain: {str(e)}")
        return None

# Function to process user input, fetch relevant documents, and generate a response
def user_input(user_question, chat_history):
    try:
        # Load FAISS vector store with OpenAI embeddings for document retrieval
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Build a combined query from the recent conversation history
        recent_context = "\n".join([f"Human: {chat['question']}\nAI: {chat['answer']}" for chat in chat_history[-5:]])
        combined_query = f"{recent_context}\nHuman: {user_question}"

        # Perform similarity search in the FAISS vector store
        docs = new_db.similarity_search(combined_query, k=7)

        # Aggregate the content of the most relevant documents
        context = "\n".join([doc.page_content for doc in docs])

        # Track the source of the documents (filenames)
        global source_tracking
        source_tracking = [doc.metadata.get('source', 'Unknown Source') for doc in docs]

        # Retrieve the conversational chain (question-answering logic)
        chain = get_conversational_chain()

        if chain is None:
            return "Sorry, something went wrong with the AI model."

        # Generate a response based on the documents and context
        response = chain.invoke(
            {
                "input_documents": docs,
                "question": combined_query,
                "chat_history": recent_context,
                "context": context,
            },
            return_only_outputs=True
        )

        # Return the generated response, or a default message if the response is empty
        return response["output_text"].strip() if response.get("output_text") else "I'm sorry, I don't have that information."
    
    except Exception as e:
        logging.error(f"Error in user_input: {str(e)}")
        return "Sorry, something went wrong with your request."

# Routes

# Route to serve the home page (HTML template for user interaction)
@app.route('/')
def home():
    try:
        return render_template('rag.html')  # Render the HTML page (rag.html should be created in templates)
    except Exception as e:
        logging.error(f"Error loading home page: {str(e)}")
        return "Sorry, something went wrong while loading the page."

# Route to handle chat interactions
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Retrieve user input and chat history from the form data
        user_message = request.form['user_input']
        chat_history = request.form.get('chat_history', [])

        # Convert chat history from string (if it's in string format) to a list of dictionaries
        if isinstance(chat_history, str):
            chat_history = eval(chat_history)

        # Get the bot's response based on user input and chat history
        bot_message = user_input(user_message, chat_history)

        # Append the new user question and bot response to the chat history
        chat_history.append({"question": user_message, "answer": bot_message})

        # Include source tracking in the response data
        response_data = {
            "response": bot_message,
            "chat_history": chat_history,
            "sources": source_tracking  # Track and return the sources (PDF filenames)
        }

        # Return the response as JSON
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Error in /chat route: {str(e)}")
        return jsonify({"response": "Sorry, an error occurred while processing your message."})

# Main entry point for the application
if __name__ == '__main__':
    try:
        # Define the path to the folder containing PDF files
        pdf_folder_path = os.path.join(os.getcwd(), 'data')  # Assuming the 'data' folder exists
        # Create the FAISS vector store using the PDF folder data
        vector_store = create_vector_store(pdf_folder_path)

        if vector_store is None:
            raise Exception("Failed to create vector store.")

        # Start the Flask web application
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error starting the application: {str(e)}")
