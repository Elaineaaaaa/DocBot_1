import streamlit as st
import os
import pandas
from dotenv import load_dotenv
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
    This app is an LLM-powered chatbot, you can ask questions regarding your loaded files.
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model 
    ''')
    st.write('Made by Zuyao')

def main():
    st.header("DocBot - Chat with your Files")
    load_dotenv()
    uploadedFiles = st.file_uploader("Upload your files", type=['pdf','csv','xlsx','.xls'], accept_multiple_files = True)
    # Upload file
    text = ""
    for file in uploadedFiles:
        extension = file.name[len(file.name)-3:]
        if(extension == 'pdf'):
            file_reader = PdfReader(file)
            for page in file_reader.pages:
                text += page.extract_text()
        elif(extension == "csv"):
            file_reader = pandas.read_csv(file)
            text += "\n".join(
                file_reader.apply(lambda row: ', '.join(row.values.astype(str)), axis=1))
        elif(extension == "lsx" or extension == "xls"):
            file_reader = pandas.read_excel(file)
            text += "\n".join(
                file_reader.apply(lambda row: ', '.join(row.values.astype(str)), axis=1))
        
    if(uploadedFiles and text):
        st.success("Successfully uploaded files")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        #Accept user questions
        query = st.text_input("Ask question about your PDF file:")

        if query:
            k = 10  # Number of nearest neighbors to retrieve
            distances = []  # List to store the distances
            labels = []
            docs = VectorStore.similarity_search(
                query=query, k=k, distances=distances, labels=labels)

            llm = ChatOpenAI(temperature=0.07, model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.divider()
            st.subheader("Answer: ")

            st.write(response)
            st.divider()


if __name__ == '__main__':
    main()


