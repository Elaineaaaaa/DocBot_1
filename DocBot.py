import streamlit as st
import os
import requests
import pandas
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Sidebar contents
with st.sidebar:
    st.title('DocBot')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot that helps you interact with documents. Ask questions about the content of your loaded files.
    - Learn more at:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models)
    ''')
    st.write('Created by Zuyao')

def main():
    st.header("Welcome to DocBot - Your Personal File Assistant")
    
    # Dynamic greeting based on the time of day
    current_hour = datetime.now().hour
    greeting = "Good morning! 😃" if current_hour < 12 else "Good afternoon! 😊" if current_hour < 18 else "Good evening! 🌙"
    st.subheader(greeting)

    # Layout for API key, model selection, and file upload
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox("Choose OpenAI Model", ["gpt-3.5-turbo", "davinci", "curie", "babbage"])
    with col2:
        uploaded_files = st.file_uploader("Upload your documents", type=['pdf', 'csv', 'xlsx', 'xls'], accept_multiple_files=True)

    # Environment variable setting for OpenAI API key
    openai_key = st.text_input("Enter your OpenAI API key:", type="password", help="This key stays private and is necessary for processing your files.")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    else:
        st.warning("API key is required to proceed.")
        return  # Stop execution if no API key is provided

    # Processing files
    text = ""
    if uploaded_files:
        for file in uploaded_files:
            extension = file.name.split('.')[-1].lower()
            if extension == 'pdf':
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or " "
            elif extension in ['csv', 'xlsx', 'xls']:
                df = pandas.read_excel(file) if extension in ['xlsx', 'xls'] else pandas.read_csv(file)
                text += "\n".join(df.apply(lambda x: ', '.join(x.values.astype(str)), axis=1))

        if text:
            st.success("Files successfully uploaded and processed!")

    # Text splitting and vector store creation
    if text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)
        embeddings = OpenAIEmbeddings(model_name=model_name)  # Use the selected model for embeddings
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # Question and Answer interface
        query = st.text_input("Ask a question about your documents:", placeholder="Type your question here...")
        if query:
            llm = ChatOpenAI(temperature=0.07, model_name=model_name)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=vector_store.similarity_search(query, k=10), question=query)
            st.subheader("Answer:")
            st.write(response)

def send_file(file):
    response = requests.post("http://127.0.0.1:8000/process-files/", files={"file": file.getvalue()})
    return response.json()

def get_answer(query, model_name):
    response = requests.post("http://127.0.0.1:8000/ask-question/", json={"query": query, "model_name": model_name})
    return response.json()


if __name__ == '__main__':
    main()
