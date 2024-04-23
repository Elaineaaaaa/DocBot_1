import streamlit as st
import os
import pandas as pd
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
    This app is an LLM-powered chatbot. You can ask questions about your loaded files.
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model 
    ''')
    st.write('Made by Zuyao')

def main():
    st.header("Welcome to DocBot - Your Personal File Assistant")
    
    # Dynamic greeting based on the time of day
    current_hour = datetime.now().hour
    if current_hour < 12:
        st.subheader("Good morning! 😃")
    elif current_hour >= 18:
        st.subheader("Good evening! 🌙")
    else:
        st.subheader("Good afternoon! 😊")
    
    openaikey = st.text_input("Enter your OpenAI API key to get started:", type="password", help="This key stays private and is necessary for processing your files.")
    if not openaikey:
        st.warning("Please enter your OpenAI API key to proceed.")
        return  # Stop execution

    os.environ["OPENAI_API_KEY"] = openaikey

    # Initialize variables
    documents = []
    text = ""
    
    # Create a List of Documents from all of our files in the ./docs folder
    docs_folder = "docs"
    if not os.path.exists(docs_folder):
        st.error("The documents folder does not exist.")
        return

    for file_name in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, file_name)
        extension = file_name.split('.')[-1].lower()
        if extension == 'pdf':
            with open(file_path, "rb") as f:
                file_reader = PdfReader(f)
                for page in file_reader.pages:
                    text += page.extract_text() or " "  # Handle pages where text extraction fails
        elif extension in ["csv", "xlsx", "xls"]:
            df = pd.read_csv(file_path) if extension == "csv" else pd.read_excel(file_path)
            text += "\n".join(df.apply(lambda row: ', '.join(map(str, row.values)), axis=1))
    
    if text:
        st.success("Files successfully uploaded and processed!")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        query = st.text_input("What would you like to know about your documents?", placeholder="Type your question here...")
        if query:
            distances, labels = [], []
            docs = VectorStore.similarity_search(query=query, k=10, distances=distances, labels=labels)
            llm = ChatOpenAI(temperature=0.07, model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.divider()
            st.subheader("Here's what I found:")
            st.write(response)
            st.write("Feel free to ask another question!")

if __name__ == '__main__':
    main()


