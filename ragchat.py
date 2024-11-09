from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Import from langchain-openai
import openai
import os
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  # Import load_dotenv to load environment variables
import logging
from langchain.callbacks import StreamingStdOutCallbackHandler
import datetime
import regex as re
import pytz
# Load environment variables from a .env file
load_dotenv()

# Set your OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set your API key in the .env file.")

# Initialize OpenAI API with the appropriate API key
openai.api_key = OPENAI_API_KEY

# Helper Functions
def get_pdf_text_from_folder(pdf_folder):
    """Extracts and combines text content from all PDFs in a folder."""
    text = ""
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Loading PDF: {pdf_path}")  # Debugging line to check PDF paths
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            if not pages:  # If no pages are found, print a message
                print(f"No pages extracted from {pdf_path}")
            for page in pages:
                print(f"Extracted content from page: {page.page_content[:100]}...")  # Show first 100 characters
                text += page.page_content + "\n\n"
    if not text.strip():  # After loading all PDFs, check if text is still empty
        print("No text extracted from any PDFs.")
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(pdf_folder):
    """Creates a FAISS vector store from text chunks using OpenAI embeddings."""
    # Extract text from PDF files
    text = get_pdf_text_from_folder(pdf_folder)
    
    # Check if text extraction was successful
    if not text.strip():
        raise ValueError(f"No text extracted from PDFs in folder: {pdf_folder}")
    
    # Split the extracted text into chunks
    text_chunks = get_text_chunks(text)
    
    # Create FAISS vector store
    try:
        # Use OpenAIEmbeddings from langchain-openai
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        print("FAISS vector store created and saved.")
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return None
    
    return vector_store



def get_conversational_chain():
    prompt_template = """ 
  

**Context:**
{context}

**User Question:**
{question}

**Assistant Response:**
"""
    model = ChatOpenAI(model="gpt-4",api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=3000)

    try:
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logging.error(f"Error loading QA chain: {e}")
        print(f"Error setting up the conversation chain. Please check the log for details.")
        return None
def user_input(user_question):
    try:
        greetings = ["hello", "hai", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if user_question.lower() in greetings:
            yield from get_greeting().split()
            return

        embeddings = OpenAIEmbeddings()
        db2 = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = db2.similarity_search(user_question,k=2)
        streaming_handler = StreamingStdOutCallbackHandler()

        chain = get_conversational_chain()
        response = chain.invoke(
            {"input_documents": docs, "question": user_question},
            callbacks=[streaming_handler],
            return_only_outputs=True
        )
        if not response or not response.get("output_text"):
            yield "Apologies, I am unable to find the answer. Can you please rephrase your question?"
        else:
            formatted_response = response["output_text"].strip()
            yield formatted_response
    except Exception as e:
        logging.error(f"Error in user_input function: {e}")
        yield f"Sorry, something went wrong. Please try again later. Error: {str(e)}"

def get_greeting():
    current_time = get_indian_time()
    current_hour = current_time.hour
    if current_hour < 12:
        return "Good morning!"
    elif 12 <= current_hour < 16:
        return "Good afternoon! "
    elif 16 <= current_hour < 18:
        return "Good evening!"
    else:
        return "Hey...  Ask questions "
    
def get_indian_time():
    india_tz = pytz.timezone('Asia/Kolkata')
    return datetime.datetime.now(india_tz)

def format_response(response):
    response = response.replace(' - ', ': ').replace('•', '*')
    response = re.sub(r'(\d+)', r'\n\1.', response)
    response = re.sub(r'\n\s*\n', '\n', response)
    return response.strip()


user_query = "What is transformer"
response = user_input(user_query)

# Print the response
for resp in response:
    print(resp)

