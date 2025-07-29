import os
import streamlit as st   #For the User Interface
st.set_page_config(page_title="Youtube Chatbot", page_icon="💬", layout="centered")
from langchain_huggingface import HuggingFaceEmbeddings # Load the  embedding model from huggingface
from langchain_chroma import Chroma #Vectorstore to store the embedded vectors
from langchain_community.document_loaders.csv_loader import CSVLoader #To load the csv file (data containing companys faq)
from langchain_community.tools import DuckDuckGoSearchRun #Search user queries Online
from langchain_groq import ChatGroq  #Load the open source Groq Models
from langgraph.graph import StateGraph, START, END #Define the State for langgraph
from langgraph.prebuilt import ToolNode,tools_condition #specialized node designed to execute tools within our workflow.
from langchain_core.messages import AnyMessage #Human message or Ai Message
from langgraph.graph.message import add_messages  ## Reducers in Langgraph ,i.e append the messages instead of replace
from typing_extensions import Annotated,TypedDict #Annotated for labelling and TypeDict to maintain graph state 
from langchain_core.tools import tool
from langchain_core.messages import trim_messages # Trim the message and keep past 2 conversation
from langgraph.checkpoint.memory import MemorySaver #Implement langgraph memory
from langchain_core.messages import HumanMessage, AIMessage
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from collections import defaultdict
from datetime import timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv  #Load environemnt variables from .env
load_dotenv()
# Create a Unique Id for each user conversation
import uuid

def main():
    # --- Initialize Components such as llm ,embedding model and DuckDUCKSearch ---
    @st.cache_resource
    def init_components():
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2") #Load the hf Embedding model
        llm = ChatGroq(temperature=0.4, model_name='Llama-3.3-70b-versatile',max_tokens=3000) #Initialize the llm
        # llm = ChatOpenAI(model='gpt-4o-mini')
        search = DuckDuckGoSearchRun()  #Duckducksearch
        return embedding_model, llm, search
    
    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("YouTube Video Configuration")
        
        # YouTube URL input
        youtube_url = st.text_input(
            "Enter YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the full YouTube video URL here"
        )
        
        # Language selection
        video_language = st.selectbox(
            "Select Video Language:",
            options=['en', 'hi'],
            index=0,
            help="Select the primary language of the video"
        )
        
        st.markdown("---")
        st.caption("Note: Changing these settings will reload the video transcript")

        st.write(youtube_url , video_language)

    # --- Create or Load Vectorstore ---
    # @st.cache_resource
    def get_vectorstore(_embedding_model, youtube_url=None, video_language='en'):
        if youtube_url:
            video_id = youtube_url.split("v=")[-1].split("&")[0]
        else:
            video_id = "default"

        persist_dir = f"chroma_index/{video_id}"  # Use unique folder per video
        vectorstore = Chroma(collection_name = video_id,
            embedding_function=_embedding_model)

        if os.path.exists(persist_dir):
            return Chroma(persist_directory=persist_dir, embedding_function=_embedding_model)
        else:
            docs = []
            documents = []
            if youtube_url:
                try:
                    ytt_api = YouTubeTranscriptApi()
                    try:
                        docs = ytt_api.fetch(video_id,languages=[video_language])
                        # print(docs)
                    except Exception as e:
                        print("Transcript not found for this video .")
                    
                    # Step 1: Group snippets by minute
                    minute_chunks = defaultdict(list)
                    minute_starts = {}

                    for snippet in docs:
                        minute = int(snippet.start // 120)
                        minute_chunks[minute].append(snippet.text)
                        
                        # Save the earliest start time per minute
                        if minute not in minute_starts:
                            minute_starts[minute] = snippet.start

                    # Step 2: Create LangChain Document objects with HH:MM:SS timestamps
                    

                    for minute in sorted(minute_chunks.keys()):
                        content = " ".join(minute_chunks[minute])
                        
                        # Format start time to HH:MM:SS
                        seconds = int(minute_starts[minute])
                        timestamp = str(timedelta(seconds=seconds))

                        metadata = {
                            "minute": minute,
                            "start_timestamp": timestamp
                        }
                        
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)

                    print(documents)
                    vectorstore.add_documents(documents)
                    
            
                except Exception as e:
                    st.sidebar.info(f"Transcript for this Video doesnot exist . Try another YouTube URL .")
                    return None
        
        return vectorstore

        
    # @st.cache_resource
    def create_retriever(_model,_vectorstore):
    
        metadata_field_info = [
        AttributeInfo(
            name="start_timestamp",
            description="The starting second of the video chunk (in format of : 'HH:MM:SS)",
            type="integer"  # Changed from string to integer
        ),
        AttributeInfo(
            name="minute",
            description="Time of the video played",
            type="string"
        )]
        # First get the base retriever from your vectorstore with increased k
        base_vectorstore_retriever = _vectorstore.as_retriever(
            search_type = "mmr",
            search_kwargs={"k": 4 , "lambda_mult":0.3} 
        )
        document_content_description = "Transcript of a youtube video"
        retriever = SelfQueryRetriever.from_llm(
            _model,
            _vectorstore,
            document_content_description,
            metadata_field_info,
            base_retriever = base_vectorstore_retriever,
            verbose=True,
            # search_kwargs={"k": 8} 
        )
        return retriever
   
        
    # --- Initialize the components i.e llm,embedding model ,vectorstore ---
    embedding_model, model, search = init_components() 
    vectorstore = get_vectorstore(embedding_model,youtube_url,video_language)

    if vectorstore is None:
        st.sidebar.info("Failed to initialize vector store. Please check the YouTube URL and try again.")
        return
    
    retriever = create_retriever(model,vectorstore)

    # Tool A. VectorStore Retriever tool (Convert the rag_chain into a tool)
    @tool
    def retrieve_vectorstore_tool(query: str) -> str:
        """
    Use this tool when the user asks about:
    - Content of a YouTube video
    - Any queries specifically about a YouTube video
    - Requests for summaries, timestamps, blog generation, etc.
    
    Input should be the exact user query.
    This tool performs a vectorstore search using a retriever and generates an answer using the provided LLM.
    """
        try:
            result = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in result])
        
        except Exception as e:
            st.info("Error occured during retrieval . {e}")
        
        print(context)
        return str(context)

    #DuckDuckSeach Tool
    @tool
    def duckducksearch_tool(query: str) -> str:
        """Use this tool Only when:
    - The question is about the current news, affairs etc.
    
    Input should be the exact search query.
    The tool will perform a web search using DuckDuckGo.
    """
        result = search.invoke(query)
        return str(result) 


    # --- Tools and bind the tools with llm ---
    tools = [retrieve_vectorstore_tool, duckducksearch_tool]
    llm_with_tools = model.bind_tools(tools=tools)

    # Initialize the StateGraph
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages] #List of messages appended

    #Function that decides which tool to use for serving the userquery
    def tool_calling_llm(state:State)->State:
        # print(state['messages'])
        selected_msg = trim_messages(
            state["messages"],
            token_counter=len,  
            max_tokens=5,  
            strategy="last",        
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
        return {"messages":[llm_with_tools.invoke(selected_msg)]}

    # Initialize the StateGraph
    builder = StateGraph(state_schema=State)

    #Adding Nodes
    builder.add_node('tool_calling_llm',tool_calling_llm) #returns the tools that is to be used
    builder.add_node('tools',ToolNode(tools=tools)) #Uses the tool specified to fetch result

    #Adding Edges
    builder.add_edge(START,'tool_calling_llm')
    builder.add_conditional_edges(
        'tool_calling_llm',
        # If the latest message from AI is a tool call -> tools_condition routes to tools
        # If the latest message from AI is a not a tool call -> tools_condition routes to LLM, then generate final response and END
        tools_condition
    )
    builder.add_edge('tools','tool_calling_llm')
    memory = MemorySaver()

    #Compile the graph
    graph = builder.compile(
        checkpointer=memory
    )

    # Initialize thread_id(unique id for each conversation) in session_state if not exists
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
    {
        'role': 'assistant',
        'content': """You are a helpful assistant with access to two tools:

1. **retrieve_vectorstore_tool**: Use this when the user asks about the content of a YouTube video. This tool uses the video transcript to answer questions. Only use the transcript to generate responses.
   - Summarize using bullet points or short paragraphs.
   - If the user asks for a summary, provide a brief overview.
   - If the answer is not in the transcript, say: "The transcript does not include that information."

2. **duckducksearch_tool**: Use this for general queries like current news, events, or anything not related to the video.

Only use a tool when needed. Keep responses accurate and easy to understand."""
    }
]

        
    st.title("Youtube ChatAssistant")

    # display enitire chat messages
    # Display entire chat messages, excluding the first assistant (system-like) message
    for i, message in enumerate(st.session_state.messages):
        if i == 0 and message["role"] == "assistant":
            continue  # Skip the first assistant message (system-level)
        
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("Your question"):
        # adding user message/query to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt) 
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Creating a conversation between human and AI
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # Invoke the graph with full message history(i.e human and ai message)
                try:
                    response = graph.invoke(
                        {"messages": langchain_messages}, #Pass the entire chat history 
                        config=config
                    )
                    final_response = response['messages'][-1].content #last msg from AI which is the final response for user's query
                except Exception as e:
                    final_response = f"There was a problem this time, please try again. {e}"  #Incase LLM fails to answer, even after using the tools 
            
                st.markdown(final_response)
        
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()