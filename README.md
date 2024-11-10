# RAG-Based Chatbot with Document Search

This is a **web-based conversational AI** application that uses **Retrieval-Augmented Generation (RAG)** for answering questions based on documents (e.g., PDFs) stored in a local directory. The application uses **OpenAI's GPT model** for generating responses and integrates **FAISS** for efficient document similarity search. It provides an interactive chat interface and retrieves relevant documents to assist with answering user queries.

---

## 📑 Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Environment Variables](#environment-variables)
5. [Running the Application](#running-the-application)
6. [Folder Structure](#folder-structure)
7. [Contributing](#contributing)
8. [License](#license)

---

## ✨ Features

- **Conversational AI**: Responds to user queries using context from uploaded documents.
- **Document-based Question Answering**: Uses **FAISS** to search for relevant documents and generates responses based on those documents.
- **History Tracking**: Maintains conversation history for personalized interactions.
- **Source Tracking**: Displays the source of each document used in generating answers (i.e., the PDF filename).
- **Interactive Web Interface**: A simple web-based interface powered by **Flask** to interact with the AI model.

---

## ⚙️ Prerequisites

Before setting up the application, ensure you have the following installed:

- **Conda** (for managing environments)
- **Python 3.7+**
- **pip** (Python package installer)
- **Flask** (for the web interface)
- **OpenAI API Key** (for interacting with OpenAI's GPT model)
- **FAISS** (for vector store and document similarity search)
- **LangChain** (for document processing and question answering)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/Sharafunneesa-pp/ragchatbot-assist.git
cd ragchatbot-assist
