import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Improved Custom CSS and HTML templates
css = '''
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    margin-top: 20px;
}

.chat-message {
    display: flex;
    align-items: flex-start;
    margin: 10px 0;
    padding: 15px;
    border-radius: 10px;
    max-width: 70%;
    word-wrap: break-word;
}

.chat-message.user {
    background-color: #2b313e;
    color: white;
    align-self: flex-end;
}

.chat-message.bot {
    background-color: #475063;
    color: white;
    align-self: flex-start;
}

.chat-message .avatar {
    margin-right: 15px;
}

.chat-message .avatar img {
    border-radius: 50%;
    width: 50px;
    height: 50px;
    object-fit: cover;
}

.chat-message .message {
    max-width: calc(100% - 70px);
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Bot">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png" alt="User">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    load_dotenv()  # Ensure the environment variable is loaded before using it
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.markdown(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    chat_container = st.empty()

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        response = user_input(user_question)
        chat_container.markdown(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
        chat_container.markdown(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                get_vector_store(text_chunks)
                st.success("Processing completed")

if __name__ == '__main__':
    main()

