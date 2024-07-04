import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_text(url):
    handle = url.split("=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(handle)
    text = ""
    for i in transcript:
        text += " " + i["text"]
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_text(text)
    return splits

def get_retriever(chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(chunks,embedding)
    return vector_db.as_retriever()

def get_prompt():
    
    prompt = """
    You are a well learned professional that is an expert in question-answering tasks,
    given the below context you will be able to draw conclusions, draw learnings and answer
    the question given 


    context = {context}
    """

    template = ChatPromptTemplate.from_messages(
        [
            ("system",prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )
    return template

def process_url(url):
    text = get_text(url)
    chunks = get_chunks(text)
    retriever = get_retriever(chunks)
    return retriever



llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
prompt = get_prompt()
qna_chain = create_stuff_documents_chain(llm,prompt)

def get_history(retriever):
    relate_system = """
    Given a chat history and the latest user question which might reference
    context in the chat history, formulate a standalone question which can be 
    understood without the chat history. Do not answer the question, just
    reformulate it if needed and otherwise return it as is
    
    
    """
    relate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",relate_system),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,relate_prompt)
    return history_aware_retriever

st.header("Youtube Chat")
with st.sidebar:
    st.subheader("Upload Url")
    url = st.text_input("Enter url", key = "url")
    process = st.button("Process", key = "process")

    if(process):
        ret = process_url(url)
        retriever = get_history(ret)
        st.session_state.rag = create_retrieval_chain(retriever,qna_chain)
        st.success('Done!')

st.subheader("Ask Question")
input = st.text_input("Enter Your Question", key = "input")
submit = st.button("submit")
if(submit):
    #st.session_state.chat_history.append(("human", input))
    response = st.session_state.rag.invoke({"input": input, "chat_history": st.session_state.chat_history})
    #st.session_state.chat_history.append(("assistant", response["answer"]))
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=input),
            AIMessage(content=response['answer']),
        ]
    )
    st.write(response["answer"])

if st.session_state.chat_history:
    st.write("Chat History:")
    for message in st.session_state.chat_history:
        role = "Human" if isinstance(message, HumanMessage) else "AI"
        st.write(f"{role.capitalize()}: {message}")