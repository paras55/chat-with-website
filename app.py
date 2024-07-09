import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



openAI_api_key = st.secrets["OPENAI_API_KEY"]

def get_vectorstore_from_urls(urls):
    # Load the webpages
    loaders = [WebBaseLoader(url) for url in urls]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)
    
    # Create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(api_key=openAI_api_key))
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(api_key=openAI_api_key, model="gpt-3.5-turbo")
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI(api_key=openAI_api_key, model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "You are an AI assistant that answers questions based on the provided context from multiple websites. Always strive to give accurate and helpful answers. If the information is not in the context, say so politely."),
      ("system", "Context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# App config
st.set_page_config(page_title="Chat with multiple websites", page_icon="🤖")
st.title("Chat with multiple websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Sample links
    sample_urls = [
        "https://dev.to/alexr/amazon-software-engineer-levels-roles-and-expectations-with-salary-1017",
        "https://www.levels.fyi/blog/scaling-to-senior-engineer.html"
    ]
    if st.button("Try with sample links"):
        st.session_state.website_urls = sample_urls

    # Text area for website URLs
    website_urls = st.text_area("Enter website URLs (one per line)", value='\n'.join(st.session_state.get('website_urls', [])))
    
    if st.button("Process Websites"):
        urls = [url.strip() for url in website_urls.split('\n') if url.strip()]
        st.session_state.website_urls = urls
        with st.spinner("Processing websites..."):
            st.session_state.vector_store = get_vectorstore_from_urls(urls)
        st.success(f"Processed {len(urls)} website(s) successfully!")

if 'website_urls' not in st.session_state or not st.session_state.website_urls:
    st.info("Please enter website URLs or click the 'Try with sample links' button, then click 'Process Websites'")

elif 'vector_store' not in st.session_state:
    st.warning("Please click 'Process Websites' to start")

else:
    # Display processed websites
    st.write("Processed websites:")
    for url in st.session_state.website_urls:
        st.write(f"- {url}")

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you with information from these websites?"),
        ]

    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        with st.spinner("Generating response..."):
            response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)