import streamlit as st
import os
import pandas
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader

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
    elif current_hour < 18:
        st.subheader("Good afternoon! 😊")
    else:
        st.subheader("Good evening! 🌙")
    
    st.write("Upload your documents, and I'll help you understand them.")

    openaikey = st.text_input("Enter your OpenAI API key to get started:", type="password", help="This key stays private and is necessary for processing your files.")
    os.environ["OPENAI_API_KEY"] = openaikey

    if not openaikey:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()

    uploadedFiles = st.file_uploader("Upload your PDF or CSV files:", type=['pdf', 'csv', 'xlsx', '.xls'], accept_multiple_files=True)
    if not uploadedFiles:
        st.info("Awaiting file uploads. Please upload any PDF or CSV files you'd like to query.")
        return

    text = ""
    for file in uploadedFiles:
        extension = file.name[len(file.name)-3:]
        if (extension == 'pdf'):
            file_reader = PdfReader(file)
            for page in file_reader.pages:
                text += page.extract_text() or " "  # Handle pages where text extraction fails
        elif(extension == "csv"):
            file_reader = pandas.read_csv(file)
            text += "\n".join(
                file_reader.apply(lambda row: ', '.join(row.values.astype(str)), axis=1))
        elif(extension == "lsx" or extension == "xls"):
            file_reader = pandas.read_excel(file)
            text += "\n".join(
                file_reader.apply(lambda row: ', '.join(row.values.astype(str)), axis=1))


    if (uploadedFiles and text):
        st.success("Files successfully uploaded and processed!")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
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
