import os
import streamlit as st   #For the User Interface
st.set_page_config(page_title="HR Policy Chatbot", page_icon="💬", layout="centered")
from langchain_huggingface import HuggingFaceEmbeddings # Load the  embedding model from huggingface
from langchain_chroma import Chroma #Vectorstore to store the embedded vectors
from langchain_community.document_loaders.csv_loader import CSVLoader #To load the csv file (data containing companys faq)
from langchain_community.tools import DuckDuckGoSearchRun #Search user queries Online
from langchain.prompts import PromptTemplate #Create a template
from langchain.chains.combine_documents import create_stuff_documents_chain #form a final prompt with 'context' and ;query'
from langchain.chains import create_retrieval_chain # "Combines a retriever (to fetch docs) with the 'create_stuff_document_chain' to automate end-to-end retrieval + answering."
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
from langchain_community.document_loaders import YoutubeLoader # Load the Youtube Transcript
from langchain_community.document_loaders.youtube import TranscriptFormat # To Get transcripts as timestamped chunks
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from dotenv import load_dotenv  #Load environemnt variables from .env
load_dotenv()
# Create a Unique Id for each user conversation
import uuid

def main():
    # --- Initialize Components such as llm ,embedding model and DuckDUCKSearch ---
    @st.cache_resource
    def init_components():
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #Load the hf Embedding model
        llm = ChatGroq(temperature=0.4, model_name='Qwen-Qwq-32b',max_tokens=650) #Initialize the llm
        search = DuckDuckGoSearchRun()  #Duckducksearch
        return embedding_model, llm, search

    # --- Create or Load Vectorstore ---
    @st.cache_resource
    def get_vectorstore(_embedding_model):
        persist_dir = "chroma_index" #Location to store embedding 

        if os.path.exists(persist_dir):
            return Chroma(persist_directory=persist_dir, embedding_function=_embedding_model)
        else:
            try:
                loader = YoutubeLoader.from_youtube_url(
                    "https://www.youtube.com/watch?v=J5_-l7WIO_w&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=17",
                    language=["hi"],
                    translation="en",
                    transcript_format=TranscriptFormat.CHUNKS,
                    chunk_size_seconds=60,
                )
                docs = loader.load()

                if docs:
                    st.write(f"Successfully loaded {len(docs)} transcript chunks")
                else:
                    st.write("No transcript data was loaded (empty result)")

            except Exception as e:
                st.write(f"Error loading YouTube transcript: {str(e)}")  
            vectorstore = Chroma.from_documents(documents=docs, embedding=_embedding_model, persist_directory=persist_dir) #Create a vector embeddings
            # vectorstore.persist() #Save the vector store
            return vectorstore
        
    @st.cache_resource
    def create_retriever(_model,_vectorstore):
    
        metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The link of the video",
            type="string"
        ),
        AttributeInfo(
            name="start_seconds",
            description="The starting second of the video chunk (in seconds as integer)",
            type="integer"  # Changed from string to integer
        ),
        AttributeInfo(
            name="start_timestamp",
            description="Human-readable timestamp (HH:MM:SS format)",
            type="string"
        )]
        # First get the base retriever from your vectorstore with increased k
        base_vectorstore_retriever = _vectorstore.as_retriever(
            # search_type = "mmr",
            search_kwargs={"k": 12,'lambda_mult':0.5}  # Increase this number as needed
        )
        document_content_description = "Transcript of a youtube video"
        retriever = SelfQueryRetriever.from_llm(
            _model,
            _vectorstore,
            document_content_description,
            metadata_field_info,
            base_retriever = base_vectorstore_retriever,
            verbose=True,
            search_kwargs={"k": 12}  # Increase this number as needed
        )
        return retriever
   
        
    # --- Initialize the components i.e llm,embedding model ,vectorstore ---
    embedding_model, model, search = init_components() 
    vectorstore = get_vectorstore(embedding_model)
    retriever = create_retriever(model,vectorstore)

         # --- RAG Chain Response ---
    # def get_rag_response(query):
    #     result = retriever.invoke(query)
    #     return result

    

    # Tool A. VectorStore Retriever tool (Convert the rag_chain into a tool)
    @tool
    def retrieve_vectorstore_tool(query: str) -> str:
        """Use this tool when the user ask about:
    - content of the youtube video
    - Any queries specifically about the youtube video 
    - If the user query involves providing summary , or about specific time stamp ,blog etc
    Input should be the exact search query.
    The tool will perform a vectorstore search using retriever."""
        return retriever.invoke(query)
        
        

    #DuckDuckSeach Tool
    @tool
    def duckducksearch_tool(query: str) -> str:
        """Use this tool Only when:
    - The question is about the current news, affairs etc.
    
    Input should be the exact search query.
    The tool will perform a web search using DuckDuckGo.
    """
        return search.invoke(query)


    # --- Tools and bind the tools with llm ---
    tools = [retrieve_vectorstore_tool, duckducksearch_tool]
    llm_with_tools = model.bind_tools(tools=tools)

    # Initialize the StateGraph
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages] #List of messages appended

    #Function that decides which tool to use for serving the userquery
    def tool_calling_llm(state:State)->State:
        print(state['messages'])
        selected_msg = trim_messages(
            state["messages"],
            token_counter=len,  #len will count the number of messages rather than tokens
            max_tokens=10,  # allow up to 10 messages.(i.e 2-3 past conversation between human and Ai)
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
            {"role": "assistant", "content": "Hello! I'm your YouTube Assistant. Ask me content about the youtube video."}
        ]

    st.title("HR Policy Chatbot")
    st.caption("Ask about company policies or general knowledge")

    # display enitire chat messages
    for message in st.session_state.messages:
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