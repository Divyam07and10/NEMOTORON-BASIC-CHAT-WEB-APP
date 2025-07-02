# 🧠 NEMOTORON-BASIC-CHAT-WEB-APP

A Streamlit-based chatbot powered by **NVIDIA’s Nemotron** model that can interactively assist users by answering queries based on a PDF help guide. The application uses **OpenCLIP** for embeddings and **Pinecone** for vector storage and retrieval.

---

## 📌 Features

- 💬 Interactive chat interface via **Streamlit**
- 🔐 Secure environment variable loading with `dotenv`
- 📄 PDF ingestion and chunking using **LangChain**
- 🧠 Embedding generation using **OpenCLIP (ViT-B-32)**
- 🌲 Semantic search with **Pinecone vector database**
- 🛡️ Offensive language filtering for safe user interaction
- 📘 Answers limited to the context of a provided Help Guide (PDX Processing System)

---

## 📂 Project Structure

```
NEMOTORON-BASIC-CHAT-WEB-APP
├── app.py                    # Streamlit web application
├── setup_rag.py              # Script to build vector DB from PDF
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not committed)
├── chatbot                   # Virtual Environment
├── rag
├── ├── document_loader.py    # Loads and splits PDF into chunks
├── ├── vector_store.py       # Handles embedding, Pinecone setup/search
├── data
└── ├── Help-Guide.pdf        # Source document for chatbot knowledge
```

---

## 🧪 How It Works

1. The **PDF Help Guide** is loaded and split into small text chunks.
2. Each chunk is embedded using **OpenCLIP** embeddings.
3. Embeddings are uploaded to a **Pinecone** vector index.
4. User queries are matched with relevant document chunks.
5. The **NVIDIA Nemotron** model generates a context-aware answer.

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/Divyam07and10/NEMOTORON-BASIC-CHAT-WEB-APP.git
cd NEMOTORON-BASIC-CHAT-WEB-APP
```

### 2. Set Up Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # on Windows use: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add `.env` File

Create a `.env` file in the root directory with the following variables:

```env
NVIDIA_API_KEY=your_nvidia_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=nemotron-chat-index
```

---

## 🛠️ Build the Vector Store

This step is **mandatory** to make the chatbot functional.

```bash
python setup_rag.py
```

This will:
- Load `Help-Guide.pdf`
- Generate embeddings
- Upload them to Pinecone

---

## 💬 Run the Chat App

```bash
streamlit run app.py
```

The chatbot will launch in your browser at `http://localhost:8501`.

---

## 🧷 Example Usage

> **User**: "How do I create a new daily schedule?"  
> **Bot**: "To create a new daily schedule, open the app, log in, go to the 'Daily Schedule' tab, select a date and KAM, and click 'Copy from Base Roster' or choose a previous date..."

---

## ❗ Notes

- The bot is **domain-specific**: it only answers questions based on `Help-Guide.pdf`.
- If a user asks an unrelated question, the bot will kindly redirect them.
- A basic **profanity filter** ensures polite and professional interactions.

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [OpenAI Python SDK (with NVIDIA endpoint)](https://pypi.org/project/openai/)
- [Pinecone](https://www.pinecone.io/)
- [LangChain](https://www.langchain.com/)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Torch](https://pytorch.org/)

---

## 📖 Source Knowledge

The chatbot relies on a PDF document named [`Help-Guide.pdf`](./Help-Guide.pdf), which contains detailed instructions for Account Administrators using the **PDX Processing System**, including:

- Daily scheduling
- Billing & settlements
- Upload/download workflows
- Managing customers & contractors
- Generating reports
- Handling fuel prices and lookups

---

## 📞 Support

If you experience issues with the PDX system, the original guide suggests contacting:

- 📧 Email: `itsupport@pdxdelivers.com`
- ☎️ Phone: `1-610-595-3390`
- 🌐 Website: [www.pdxdelivers.com](http://www.pdxdelivers.com/)

---

## ✅ License

This project is for educational or internal use only. Respect any licenses attached to NVIDIA models, Pinecone, and other third-party libraries used herein.
