# 🤖 RAG Q&A Chatbot – Chat with Multiple PDFs

Welcome to **GenAI Doc Chat**, a Retrieval-Augmented Generation (RAG)-based chatbot powered by **Google Gemini Pro**, **FAISS**, and **LangChain**.  
Upload your PDF documents and ask contextual questions based on their content.

🔗 **Live Demo**: [genai-doc-chat.streamlit.app](https://genai-doc-chat.streamlit.app/)

---

## 📌 Features

- 📄 Upload and chat with **multiple PDF files**
- 💬 Ask questions in **natural language**
- 🤖 Powered by **Google Gemini** via **LangChain**
- 🔍 Uses **RAG (Retrieval-Augmented Generation)** with **FAISS vector store**
- 📥 Download complete **conversation history**
- 🎨 Styled chat interface with avatars

---

## 🧱 Tech Stack

| Technology     | Purpose                         |
|----------------|----------------------------------|
| Streamlit      | Web interface                   |
| LangChain      | Question-answering pipeline     |
| Google Gemini  | LLM model (via Google GenAI API)|
| FAISS          | Vector store for text chunks    |
| PyPDF2         | PDF text extraction             |
| Pandas         | Data handling & export          |

---

## 1. Clone the Repository

```bash
git clone https://github.com/Manika7777/CSI_WEEK8_MS.git
cd CSI_WEEK8_MS
```

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

## 3. Run the App

```bash
streamlit run app.py
```

---

## 🔐 Get Google GenAI API Key

1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Sign in with your Google account
3. Generate an API Key from your [API Key page](https://ai.google.dev/)
4. Paste the API Key in the app sidebar when prompted

---

## 📁 Folder Structure

```
CSI_WEEK8_MS/
├── app.py
├── requirements.txt
├── faiss_index/
```

---

## 👩‍💻 Author

**Manika Sarkar**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/manika-sarkar-264426295/) 
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/Manika7777)

---

⭐️ Don’t forget to **star this repo** if you like it!
