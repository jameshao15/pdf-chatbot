from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def main(): 
    load_dotenv()

    #load basic page setup
    st.set_page_config(page_title = 'Upload your PDF')
    st.title("PDF ChatBot 📝")   
    st.header("Upload your PDF then ask questions!")
    
    #upload file
    pdf = st.file_uploader("Upload here ⬇️")

    #read, extract, and split text
    if pdf != None: 
        pdf_reader = PdfReader(pdf)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()
            
        #split pdf into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators = '\n',
            chunk_size = 750,
            chunk_overlap = 150,
            length_function = len
        )
        chunks = text_splitter.split_text(full_text)

        #create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        #question prompt and show user input
        question = st.text_input("Ask a question about your PDF: ")
        if question:
            documents = knowledge_base.similarity_search(question)

            #create chain for easier question/answering
            llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm, chain_type='stuff')


            response = chain.run(input_documents = documents, question=question)

            st.write(response)

if __name__ == "__main__":
    main()