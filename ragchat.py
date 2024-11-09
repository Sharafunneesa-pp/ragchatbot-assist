from flask import Flask, request, render_template
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import logging
import re

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM (Language Model)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=3000)

# Define  RAG chatbot function
def chat_with_rag(message):
    try:
        # Load PDF documents
        pdf_loader = PyPDFDirectoryLoader("data")
        docs = pdf_loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
        documents = text_splitter.split_documents(documents=docs)

        # Create embeddings for the text chunks
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        db = FAISS.from_texts([doc.page_content for doc in documents], embeddings)  # Use the text content
        retriever = db.as_retriever()

        # Define the prompt template
        template = """Answer the question based only on the following context:

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Function to format documents
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        # Retrieve relevant documents for the question
        retriever_output = retriever.run({"question": message})
        formatted_docs = format_docs(retriever_output)

        # Generate the answer by passing the formatted docs and the message to the model
        response = llm({"context": formatted_docs, "question": message})
        
        # Parse and return the response
        return response['text']

    except Exception as e:
        logging.error(f"Error in chat_with_rag: {e}")
        return "Error processing the request."

# Define  Flask routes
@app.route('/')
def home():
    return render_template('ragchat.html')

# chat api end point 
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.form['user_input']
        bot_message = chat_with_rag(user_message)
        
        # Define the regex pattern to extract the answer
        pattern = r"Answer:\s*(.*)"
        match = re.search(pattern, bot_message, re.DOTALL)

        if match:
            answer = match.group(1).strip()
            print("Extracted Answer:", answer)
            return {'response': answer}
        else:
            print("Answer not found")
            return {'response': "Answer not found as per context"}
    except Exception as e:
        logging.error(f"Error in chat route: {e}")
        return {'response': "An error occurred during the chat."}

if __name__ == '__main__':
    app.run(debug=True)
