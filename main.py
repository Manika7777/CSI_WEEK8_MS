import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import asyncio
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks, model_name, api_key=None):
    ensure_event_loop()
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return embeddings


def get_conversational_chain(model_name, api_key=None):
    if model_name == "Google AI":
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in
        the context, just say "answer is not available in the context". Don't make up answers.

        Context:\n{context}\n
        Question:\n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if not api_key or not pdf_docs:
        st.warning("Please upload PDF files and provide API key before processing.")
        return

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text, model_name)
    embeddings = get_vector_store(text_chunks, model_name, api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(model_name, api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    user_question_output = user_question
    response_output = response['output_text']
    pdf_names = [pdf.name for pdf in pdf_docs]

    conversation_history.append((user_question_output, response_output, model_name,
                                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .chat-message.user {
        background-color: #1f2c3b;
    }
    .chat-message.bot {
        background-color: #2e3b4e;
    }
    .chat-message .avatar {
        flex-shrink: 0;
    }
    .chat-message .avatar img {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        border: 2px solid #ddd;
        object-fit: cover;
    }
    .chat-message .message {
        color: #fff;
        line-height: 1.6;
        font-size: 1rem;
        flex-grow: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="chat-message user">
        <div class="avatar"><img src="https://i.ibb.co/9hL3DNc/user-icon.png"></div>
        <div class="message">{user_question_output}</div>
    </div>
    <div class="chat-message bot">
        <div class="avatar"><img src="https://i.ibb.co/Wg8w6Pq/google-gemini.png"></div>
        <div class="message">{response_output}</div>
    </div>
    """, unsafe_allow_html=True)

    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history[:-1]):
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar"><img src="https://www.google.com/imgres?q=photos%20for%20%20user&imgurl=https%3A%2F%2Fw7.pngwing.com%2Fpngs%2F81%2F570%2Fpng-transparent-profile-logo-computer-icons-user-user-blue-heroes-logo-thumbnail.png&imgrefurl=https%3A%2F%2Fwww.pngwing.com%2Fen%2Fsearch%3Fq%3Duser&docid=8Qj_3LCalWAqLM&tbnid=dFU4qv6qxI532M&vet=12ahUKEwjksK3nhtuOAxVz4jgGHebZMpYQM3oECCwQAA..i&w=360&h=360&hcb=2&ved=2ahUKEwjksK3nhtuOAxVz4jgGHebZMpYQM3oECCwQAA"></div>
            <div class="message">{question}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar"><img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Ffree-photos-vectors%2Fchatbot-ai&psig=AOvVaw3XyIAi2dEfdWjJ-s2V_z3X&ust=1753637303634000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCOifpZuG244DFQAAAAAdAAAAABAE"></div>
            <div class="message">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

    if conversation_history:
        df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("⬅️ Click the **Download** button on the left to save this conversation.")
    st.snow()


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.header("🤖 RAG Q&A Chatbot (PDFs + Google GenAI + FAISS)")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    st.sidebar.markdown(
        """
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/manika-sarkar-264426295/) 
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Manika7777)
        """
    )

    model_name = st.sidebar.radio("🔍 Select the Model:", ("Google AI",))
    api_key = st.sidebar.text_input("🔑 Enter your Google API Key:", type="password")
    st.sidebar.markdown("➡️ [Get your API key here](https://ai.google.dev/)")

    with st.sidebar:
        st.title("🛠 Menu")
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset Chat")
        clear_button = col1.button("Undo Last")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.user_question = None
        if clear_button and st.session_state.conversation_history:
            st.session_state.conversation_history.pop()

        pdf_docs = st.file_uploader("📤 Upload your PDF Files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("✅ PDFs processed successfully!")
            else:
                st.warning("Please upload at least one PDF.")

    user_question = st.text_input("💬 Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""


if __name__ == "__main__":
    main()
