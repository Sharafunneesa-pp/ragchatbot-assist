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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

# Initialize global vector store
vector_store = None

# Helper Functions
def get_pdf_text_from_folder(pdf_folder):
    text = ""
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            for page in pages:
                text += page.page_content + "\n\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def create_vector_store(pdf_folder):
    text = get_pdf_text_from_folder(pdf_folder)
    text_chunks = get_text_chunks(text)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Use the provided context and chat history to answer the question accurately. If you can't answer based on the information, please state that.

    Context: {context}
    Chat History: {chat_history}
    Human: {question}
    AI: 
    """
    model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=3000)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, chat_history):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    recent_context = "\n".join([f"Human: {chat['question']}\nAI: {chat['answer']}" for chat in chat_history[-5:]])
    combined_query = f"{recent_context}\nHuman: {user_question}"
    docs = new_db.similarity_search(combined_query, k=10)
    context = "\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    response = chain(
        {
            "input_documents": docs,
            "question": combined_query,
            "chat_history": recent_context,
            "context": context,
        },
        return_only_outputs=True
    )
    return response["output_text"].strip() if response.get("output_text") else "I'm sorry, I don't have that information."

# Routes
@app.route('/')
def home():
    return render_template('ragchat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_input']
    chat_history = request.form.get('chat_history', [])
    if isinstance(chat_history, str):
        chat_history = eval(chat_history)
    bot_message = user_input(user_message, chat_history)
    chat_history.append({"question": user_message, "answer": bot_message})
    return jsonify({"response": bot_message, "chat_history": chat_history})

if __name__ == '__main__':
    pdf_folder_path = os.path.join(os.getcwd(), 'data')
    vector_store = create_vector_store(pdf_folder_path)
    app.run(debug=True)
