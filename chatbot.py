from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
#from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai
from langchain.callbacks import get_openai_callback

def main(): 
    load_dotenv()

    #load basic page setup
    st.set_page_config(page_title = 'Upload your PDF')
    st.title("PDF ChatBot 📝")   
    st.header("Upload your PDF then ask questions! 💞")
    
    #upload file
    pdf = st.file_uploader("Upload here ⬇")

    #read, extract, and split text🦄
    if pdf != None: 
        pdf_reader = PdfReader(pdf)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()
            
        #split pdf into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators = '\n',
            chunk_size = 500,
            chunk_overlap = 50,
            length_function = len
        )
        chunks = text_splitter.split_text(full_text)

        #create embeddings
        embeddings = OpenAIEmbeddings()
        knwoledge_base = FAISS.from_texts(chunks, embeddings)

        #question prompt and show user input
        question = st.text_input("🙋🏻‍♀️ Ask a question about your PDF: ")
        if question:
            documents = knwoledge_base.similarity_search(question)

            llm = openai()
            chain = load_qa_chain(llm, chain_type)

        #display costs
        with get_openai_callback() as cb:
            response = chain.run(input_documents = documents, question=question)
            print(cb)

        st.write(response)



#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

#def count_tokens(text: str) -> int:
#    return len(tokenizer.encode(text))

if __name__ == "__main__":
    main()