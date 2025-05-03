from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
import os
import time

def main(): 
    load_dotenv()

    # Load basic page setup
    st.set_page_config(page_title='ChatBot', layout="wide")
    
    # Initialize session state for tracking
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0
    if 'total_cost' not in st.session_state:
        st.session_state.total_cost = 0
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'active_mode' not in st.session_state:
        st.session_state.active_mode = "general_chat"
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'pdf_files' not in st.session_state:
        st.session_state.pdf_files = []
    if 'uploaded_pdf_names' not in st.session_state:
        st.session_state.uploaded_pdf_names = []
    
    # Define default parameter values
    default_temperature = 0.7 if st.session_state.active_mode == "general_chat" else 0.0
    default_max_tokens = 1000 if st.session_state.active_mode == "general_chat" else 500
    default_chunk_size = 750
    default_chunk_overlap = 150
    default_k_retrieval = 4
    
    # Create a sidebar for model settings and token stats
    with st.sidebar:
        st.subheader("Chat Mode")
        # Create radio buttons to select mode
        chat_mode = st.radio(
            "Select Chat Mode:",
            ["General Chat", "PDF Chat"],
            index=0 if st.session_state.active_mode == "general_chat" else 1,
            help="General Chat: Talk about anything | PDF Chat: Ask questions about specific PDFs"
        )
        
        # Set the active mode based on selection
        st.session_state.active_mode = "general_chat" if chat_mode == "General Chat" else "pdf_chat"
        
        # PDF upload section (only shown in PDF mode)
        if st.session_state.active_mode == "pdf_chat":
            st.subheader("PDF Upload")
            
            # Multiple PDF uploader
            uploaded_pdfs = st.file_uploader(
                "Upload PDFs ⬇️", 
                type="pdf", 
                accept_multiple_files=True,
                help="You can upload multiple PDF files"
            )
            
            # Display currently uploaded PDFs with remove options
            #st.subheader("Uploaded PDFs")
            
            # Process newly uploaded PDFs
            if uploaded_pdfs:
                with st.spinner("Processing PDFs..."):
                    # Get list of new PDFs that haven't been processed yet
                    new_pdfs = [pdf for pdf in uploaded_pdfs if pdf.name not in st.session_state.uploaded_pdf_names]
                    
                    if new_pdfs:
                        # Initialize the knowledge base if it doesn't exist
                        if st.session_state.knowledge_base is None:
                            # Create empty embeddings first time
                            embeddings = OpenAIEmbeddings()
                            st.session_state.knowledge_base = FAISS.from_texts(["Initial document"], embeddings)
                        
                        for pdf in new_pdfs:
                            # Add to list of processed PDFs
                            if pdf.name not in st.session_state.uploaded_pdf_names:
                                st.session_state.uploaded_pdf_names.append(pdf.name)
                                st.session_state.pdf_files.append(pdf)
                                
                                # Process the PDF
                                pdf_reader = PdfReader(pdf)
                                full_text = ""
                                for page in pdf_reader.pages:
                                    full_text += page.extract_text()
                                
                                # Use chunk settings from sidebar
                                text_splitter = RecursiveCharacterTextSplitter(
                                    separators='\n',
                                    chunk_size=default_chunk_size,
                                    chunk_overlap=default_chunk_overlap,
                                    length_function=len
                                )
                                chunks = text_splitter.split_text(full_text)
                                
                                # Add document source metadata to each chunk
                                chunks_with_metadata = []
                                for i, chunk in enumerate(chunks):
                                    chunks_with_metadata.append({
                                        "content": chunk,
                                        "metadata": {"source": pdf.name, "chunk_id": i}
                                    })
                                
                                # Add to existing knowledge base
                                embeddings = OpenAIEmbeddings()
                                new_vectorstore = FAISS.from_texts(
                                    [chunk["content"] for chunk in chunks_with_metadata],
                                    embeddings,
                                    metadatas=[chunk["metadata"] for chunk in chunks_with_metadata]
                                )
                                
                                # Merge with existing knowledge base if it exists
                                if st.session_state.knowledge_base and len(st.session_state.uploaded_pdf_names) > 1:
                                    st.session_state.knowledge_base.merge_from(new_vectorstore)
                                else:
                                    st.session_state.knowledge_base = new_vectorstore
                                    
                                success_message = st.success(f"PDF processed: {pdf.name}")
                                time.sleep(3) 
                                success_message.empty()
            
            # Display uploaded PDFs with option to remove
            if st.session_state.uploaded_pdf_names:
                pass
            else:
                st.info("No PDFs uploaded yet.")
        
        
        st.subheader("Model Settings")
        
        # Add model selection based on chat mode
        if st.session_state.active_mode == "general_chat":
            model_option = st.selectbox(
                'Select Chat Model:',
                ('gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'),
                index=0
            )
        else:
            model_option = st.selectbox(
                'Select PDF Chat Model:',
                ('gpt-3.5-turbo-instruct', 'text-davinci-003', 'text-davinci-002', 'text-curie-001'),
                index=0
            )
        
        # Create a tab interface for different parameter categories
        tabs = st.tabs(["Basic", "Retrieval"])
        
        # Basic parameters tab
        with tabs[0]:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=default_temperature, step=0.1,
                                  help="Controls randomness: 0 = deterministic, 1 = creative")
            
            max_tokens = st.slider("Max Tokens", min_value=50, max_value=4000, value=default_max_tokens, step=50,
                                 help="Maximum number of tokens to generate in the response")
        
        # Retrieval parameters tab (for PDF chat)
        with tabs[1]:
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=default_chunk_size, step=50,
                                help="Size of text chunks when splitting the document")
            
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=default_chunk_overlap, step=10,
                                   help="Amount of overlap between chunks to maintain context")
            
            k_retrieval = st.slider("Number of Chunks to Retrieve", min_value=1, max_value=10, value=default_k_retrieval, step=1,
                                 help="Number of relevant document chunks to include in the context")
        
        # Token usage statistics
        st.subheader("Token Usage Statistics")
        
        # Create placeholders for dynamic updating
        token_placeholder = st.empty()
        cost_placeholder = st.empty()
        
        # Update placeholders with current values
        token_placeholder.markdown(f"**Total Tokens Used:** {st.session_state.total_tokens}")
        cost_placeholder.markdown(f"**Total Cost (USD):** ${st.session_state.total_cost:.5f}")
        
        # Store placeholders in session state for updating
        st.session_state.token_placeholder = token_placeholder
        st.session_state.cost_placeholder = cost_placeholder

    # Main chat area
    st.title("OpenAI Chatbot")
    
    # Mode indicator with PDF count
    if st.session_state.active_mode == "general_chat":
        mode_indicator = "💬 General Chat Mode"
    else:
        pdf_count = len(st.session_state.uploaded_pdf_names)
        if pdf_count == 0:
            mode_indicator = "📑 PDF Chat Mode - No PDFs uploaded"
        elif pdf_count == 1:
            mode_indicator = f"📑 PDF Chat Mode - 1 PDF: {st.session_state.uploaded_pdf_names[0]}"
        else:
            mode_indicator = f"📑 PDF Chat Mode - {pdf_count} PDFs uploaded"
    
    st.subheader(mode_indicator)
    
    # Create a container for messages (with scrollable height)
    message_container = st.container()
    
    # Display chat history in the message container
    with message_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "tokens" in message and "cost" in message:
                    st.caption(f"Tokens: {message['tokens']} | Cost: ${message['cost']:.5f}")
                if "source_documents" in message:
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(message["source_documents"]):
                            # Handle different document formats
                            if isinstance(doc, dict) and 'metadata' in doc:
                                source = doc['metadata'].get('source', 'Unknown')
                                chunk_id = doc['metadata'].get('chunk_id', 'Unknown')
                                
                                # Check for different content field names
                                if 'page_content' in doc:
                                    content = doc['page_content']
                                elif 'content' in doc:
                                    content = doc['content']
                                else:
                                    content = 'No content available'
                            else:
                                # Fallback for unknown structure
                                source = 'Unknown'
                                chunk_id = 'Unknown'
                                content = str(doc)
                            
                            st.markdown(f"**Source {i+1}:** {source} (Chunk {chunk_id})")
                            st.text(content[:200] + "..." if len(content) > 200 else content)
    
    # Check if PDF chat is selected but no PDF is uploaded
    if st.session_state.active_mode == "pdf_chat" and not st.session_state.uploaded_pdf_names:
        st.info("Please upload at least one PDF in the sidebar to start asking questions.")
    
    # Create a separate container at the bottom for the input
    input_container = st.container()
    
    # Question/message input
    prompt_text = "Ask a question about your PDFs:" if st.session_state.active_mode == "pdf_chat" else "Send a message:"
    with input_container:
        user_input = st.chat_input(prompt_text)
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Display assistant thinking indicator
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                # Determine which mode to use
                if st.session_state.active_mode == "general_chat":
                    # General chat mode using ChatOpenAI
                    chat_model = ChatOpenAI(
                        model=model_option,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Format the conversation history
                    messages = []
                    
                    # Add system message
                    messages.append({
                        "role": "system", 
                        "content": "You are a helpful, friendly, and knowledgeable AI assistant. Respond to the user in a conversational and engaging manner."
                    })
                    
                    # Add the last few messages from chat history (limit to last 20 exchanges)
                    recent_history = st.session_state.chat_history[-20:] if len(st.session_state.chat_history) > 20 else st.session_state.chat_history
                    for msg in recent_history[:-1]:  # Exclude the latest user message as we'll add it separately
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    # Add the current user message
                    messages.append({"role": "user", "content": user_input})
                    
                    # Get response with token tracking
                    with get_openai_callback() as cb:
                        response = chat_model.invoke(messages)
                        
                        # Extract response content
                        response_content = response.content
                        
                        # Update token counts and costs
                        tokens_used = cb.total_tokens
                        cost = cb.total_cost
                        
                        st.session_state.total_tokens += tokens_used
                        st.session_state.total_cost += cost
                        
                        # Update the sidebar with new values
                        st.session_state.token_placeholder.markdown(f"**Total Tokens Used:** {st.session_state.total_tokens}")
                        st.session_state.cost_placeholder.markdown(f"**Total Cost (USD):** ${st.session_state.total_cost:.5f}")
                        
                        # Add assistant response to chat history with token info
                        assistant_response = {
                            "role": "assistant", 
                            "content": response_content,
                            "tokens": tokens_used,
                            "cost": cost
                        }
                        st.session_state.chat_history.append(assistant_response)
                        
                        # Update the placeholder with the response
                        message_placeholder.markdown(response_content)
                        st.caption(f"Tokens: {tokens_used} | Cost: ${cost:.5f}")
                
                else:
                    # PDF chat mode using the RAG framework
                    if st.session_state.knowledge_base is not None and st.session_state.uploaded_pdf_names:
                        # Get relevant documents
                        retriever = st.session_state.knowledge_base.as_retriever(search_kwargs={"k": k_retrieval})
                        
                        # Set up the prompt template with source attribution
                        system_prompt = (
                            "You are an assistant for question-answering tasks. "
                            "Use the following pieces of retrieved context to answer "
                            "the question. If you don't know the answer, say that you "
                            "don't know. Keep the answer concise and relevant to the question. "
                            "\n\n"
                            "{context}"
                        )
                        prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", system_prompt),
                                ("human", "{input}"),
                            ]
                        )
                        
                        # Create the LLM with parameters from sidebar settings
                        llm = OpenAI(
                            model_name=model_option,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        
                        # Create the question answering chain
                        question_answer_chain = create_stuff_documents_chain(llm, prompt)
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                        
                        # Get response with token tracking
                        with get_openai_callback() as cb:
                            response = rag_chain.invoke({"input": user_input})
                            
                            # Update token counts and costs
                            tokens_used = cb.total_tokens
                            cost = cb.total_cost
                            
                            st.session_state.total_tokens += tokens_used
                            st.session_state.total_cost += cost
                            
                            # Update the sidebar with new values
                            st.session_state.token_placeholder.markdown(f"**Total Tokens Used:** {st.session_state.total_tokens}")
                            st.session_state.cost_placeholder.markdown(f"**Total Cost (USD):** ${st.session_state.total_cost:.5f}")
                            
                            # Extract source documents for citation
                            source_docs = response["context"]
                            
                            # Add assistant response to chat history with token info and source docs
                            # Handle the case where source_docs might be in different formats
                            processed_source_docs = []
                            for doc in source_docs:
                                if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                                    # Document object - convert to dict for storage
                                    processed_source_docs.append({
                                        'metadata': doc.metadata,
                                        'page_content': doc.page_content
                                    })
                                else:
                                    # Already a dict or other format - store as is
                                    processed_source_docs.append(doc)
                                    
                            assistant_response = {
                                "role": "assistant", 
                                "content": response["answer"],
                                "tokens": tokens_used,
                                "cost": cost,
                                "source_documents": processed_source_docs
                            }
                            st.session_state.chat_history.append(assistant_response)
                            
                            # Update the placeholder with the response
                            message_placeholder.markdown(response["answer"])
                            st.caption(f"Tokens: {tokens_used} | Cost: ${cost:.5f}")
                            
                            # Display source documents in expandable section
                            with st.expander("View Source Documents"):
                                for i, doc in enumerate(source_docs):
                                    # Access metadata and content correctly based on document structure
                                    if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                                        # LangChain Document object
                                        source = doc.metadata.get('source', 'Unknown')
                                        chunk_id = doc.metadata.get('chunk_id', 'Unknown')
                                        content = doc.page_content
                                    elif isinstance(doc, dict) and 'metadata' in doc:
                                        # Dictionary structure
                                        source = doc['metadata'].get('source', 'Unknown')
                                        chunk_id = doc['metadata'].get('chunk_id', 'Unknown')
                                        content = doc.get('page_content', doc.get('content', 'No content available'))
                                    else:
                                        # Fallback for unknown structure
                                        source = 'Unknown'
                                        chunk_id = 'Unknown'
                                        content = str(doc)
                                    
                                    st.markdown(f"**Source {i+1}:** {source} (Chunk {chunk_id})")
                                    st.text(content[:200] + "..." if len(content) > 200 else content)
                            
                    else:
                        # No PDFs uploaded
                        message_placeholder.markdown("Please upload at least one PDF in the sidebar first.")
            
            # Force a refresh of the page
            st.rerun()

if __name__ == "__main__":
    main()