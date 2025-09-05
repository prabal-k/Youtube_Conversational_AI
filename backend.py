from IPython.display import Image, display
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
from langchain_core.documents import Document
from collections import defaultdict
from datetime import timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate ,PromptTemplate

import edge_tts
import tempfile
from pydub import AudioSegment
from playsound import playsound
import asyncio
import datetime
import os

from dotenv import load_dotenv  #Load environemnt variables from .env
load_dotenv()
# Create a Unique Id for each user conversation
import uuid
import re


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2") #Load the hf Embedding model
# llm = ChatOpenAI(model='gpt-4o-mini')
llm = ChatGroq(temperature=0.4, model_name='Llama-3.3-70b-versatile',max_tokens=3000) #Initialize the llm
search = DuckDuckGoSearchRun()  #Duckducksearch

# Global retriever (reset per new video)
retriever = None

def process_youtube_video(youtube_url: str, language: str = "en"):
    """Fetch transcript of a YouTube video and create an in-memory vectorstore."""

    global retriever
    # print(youtube_url)

    # Extract video id
    video_id = youtube_url.split("v=")[-1].split("&")[0]

    # Create a temporary vector store (in-memory, no persistence)
    vectorstore = Chroma(
        collection_name=video_id,
        embedding_function=embedding_model,
        persist_directory=None  # ensures it is NOT saved permanently
    )

    documents = []
    ytt_api = YouTubeTranscriptApi()

    try:
        # Fetch transcript
        docs = ytt_api.fetch(video_id, languages=[language])
        print("Transcript fetched successfully!")

        # Step 1: Group snippets by minute
        minute_chunks = defaultdict(list)
        minute_starts = {}

        for snippet in docs:
            minute = int(snippet.start // 60)
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

        # Add to vectorstore
        vectorstore.add_documents(documents)

        # Create retriever for QA
        metadata_field_info = [
            {"name": "start_timestamp", "description": "Video start timestamp", "type": "string"},
            {"name": "minute", "description": "Minute of video", "type": "integer"},
        ]

        base_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "lambda_mult": 0.3})

        document_content_description = "Transcript of a youtube video"
        retriever = SelfQueryRetriever.from_llm(
                    llm,
                    vectorstore,
                    document_content_description,
                    metadata_field_info,
                    base_retriever = base_retriever,
                    verbose=True,
                    # search_kwargs={"k": 8} 
                )

    except Exception as e:
        print(f"Error fetching transcript: {e}")
        retriever = None

# Audio Generation Setup
VOICES = [
    'en-AU-NatashaNeural', 'en-AU-WilliamNeural', 'en-CA-ClaraNeural', 'en-CA-LiamNeural',
    'en-GB-LibbyNeural', 'en-GB-MaisieNeural', 'en-IN-NeerjaNeural',
    "en-US-AvaNeural", "en-US-EmmaNeural", "en-US-AndrewMultilingualNeural",
    "en-US-AriaNeural", "en-US-AvaMultilingualNeural", "en-US-BrianMultilingualNeural",
    "en-US-ChristopherNeural", "en-US-EmmaMultilingualNeural", 'en-US-EricNeural'
]

VOICE = VOICES[9]  # en-US-AriaNeural

# Output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Async function to generate audio for each response
async def _generate_audio_files(text):
    """Generate unique audio file for text response."""
    # Generate unique filename based on datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(output_dir, f"audio_{timestamp}.mp3")
    
    # Clean text for better TTS synthesis
    cleaned_text = text.lower().strip()
    
    try:
        communicate = edge_tts.Communicate(cleaned_text, VOICE, rate='+0%', pitch="+1Hz")
        await communicate.save(filename)
        print(f"Audio saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def generate_audio_files(text):
    """Synchronous wrapper for audio generation."""
    return asyncio.run(_generate_audio_files(text))

from pydantic import BaseModel,Field
from typing import Literal 

class Routequery(BaseModel):
    decision : Literal["general_query","video_qa"] = Field(...,description="Given a user question choose to route to `general_query` for greetings and general conversation, or `video_qa` for video analysis and Q&A")

from typing_extensions import TypedDict
from typing import List
from langchain_core.messages import BaseMessage

class State(TypedDict):
    """Represents the state of our graph
    
    Attributes:
    messages : All the messages in our graph, including AIMessage, HumanMessage.
    audio_file : Path to generated audio file for the latest response.
    """
    messages: Annotated[list[BaseMessage], add_messages] #List of messages appended
    audio_file: str = None  # Path to the generated audio file

def is_greeting_or_general(text: str) -> bool:
    """Check if the text is a general greeting, casual inquiry, or non-video related query."""
    general_patterns = [
        r'\b(hi|hello|hey|greetings?|good\s+(morning|afternoon|evening|day))\b',
        r'\bhow\s+(are|is)\s+you\b',
        r'\bwhat\'?s\s+up\b',
        r'\bhow\s+do\s+you\s+do\b',
        r'\bnice\s+to\s+(meet|see)\s+you\b',
        r'^\s*(hi|hello|hey)\s*[!.?]*\s*$',
        r'^\s*how\s+are\s+you\s*[!.?]*\s*$',
        r'\bwho\s+are\s+you\b',
        r'\bwhat\s+can\s+you\s+do\b',
        r'\btell\s+me\s+about\s+yourself\b'
    ]
    
    text_lower = text.lower().strip()
    return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in general_patterns)

# Router Setup
system_msg_router = """You are an expert at routing user queries for a YouTube video analysis system.

Analyze the user's message and route it appropriately:

1. **general_query**: Route here if the user query is:
   - General greetings (like "hi", "hello", "how are you", etc.)
   - Casual conversation or small talk
   - Questions about the system's capabilities
   - Non-video related queries
   - General questions that don't require video transcript analysis

2. **video_qa**: Route here if the user query is:
   - Asking for video summaries or analysis
   - Specific questions about video content
   - Timestamp-related queries
   - Content explanations from the video
   - Any analytical questions about the video transcript
   - Requests for information that would be found in a video

Choose the most appropriate route based on the user's intent."""

template_route = ChatPromptTemplate([
    ("system", system_msg_router),
    ("human", "{question}")
])
structured_llm_router = llm.with_structured_output(Routequery)
chain_router = template_route | structured_llm_router

def route_query(state: State) -> dict:
    """Route question to appropriate handler.
    
    Arguments:
        state(dict) : Current graph state.

    Returns:
        dict : Update adding the intent of the user.
    """
    user_msg = next(msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))
    
    # First check if it's a simple greeting using regex
    if is_greeting_or_general(user_msg.content):
        return {"decision": "general_query"}
    
    # Use LLM for more complex routing
    intent = chain_router.invoke({"question": user_msg.content})
    return {"decision": intent.decision}

def handle_general_query(state: State) -> State:
    """Handle general queries, greetings, and casual conversation with audio generation."""
    
    messages = state["messages"]
    user_msg = next(msg for msg in reversed(messages) if isinstance(msg, HumanMessage))
    
    # Prepare conversation history for context
    conversation_history = "\n".join(
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in messages[:-1]  # Exclude current message
    )
    
    general_system_msg = """You are a friendly AI assistant specialized in YouTube video analysis and content generation. 

Your capabilities include:
- Analyzing YouTube video transcripts and providing detailed summaries
- Answering specific questions about video content with precise timestamps
- Engaging in general conversation and providing helpful information
- Assisting with various queries beyond video analysis

When responding to general queries, greetings, or casual conversation:
- Be warm, friendly, and helpful
- Provide informative and engaging responses
- If discussing your capabilities, mention your YouTube video analysis features
- Keep responses conversational and natural
- Be concise but comprehensive in your explanations
- Show enthusiasm for helping with their requests

Respond to the user's message in a natural, engaging way. Keep responses focused and helpful."""

    template_general = ChatPromptTemplate.from_messages([
        ("system", general_system_msg),
        ("human", f"Conversation History:\n{conversation_history}\n\nCurrent Message: {user_msg.content}")
    ])

    chain_general = template_general | llm
    response = chain_general.invoke({})
    
    # Generate audio for the response
    audio_file = None
    try:
        audio_file = generate_audio_files(response.content)
        print(f"Generated audio for general query: {audio_file}")
    except Exception as e:
        print(f"Error generating audio for general query: {e}")
    
    return {
        "messages": messages + [response],
        "audio_file": audio_file
    }

def handle_video_qa(state: State) -> State:
    """Handle video analysis, summarization, and Q&A with audio generation."""

    system_msg_video = """You're a helpful assistant that analyzes YouTube video transcripts and provides comprehensive answers.

Your tasks:
- Provide clear, detailed summaries or answers based on the video transcript
- Arrange content chronologically using the provided timestamps when relevant
- Answer specific questions about the video content with relevant timestamp references
- If asked for a summary, provide a well-structured overview of the main topics covered
- For specific questions, focus on the relevant sections and provide timestamp references when helpful

Guidelines:
- Always use only the transcript content for your answers
- Include relevant timestamps when discussing specific parts of the video (format: "At [timestamp]...")
- Keep your tone natural, informative, and engaging
- Provide comprehensive yet concise responses
- Return responses in English
- If the transcript doesn't contain information to answer the question, politely explain this
- Structure longer responses with clear organization
- Focus on the most important and relevant information"""

    # Get all messages (including history)
    messages = state["messages"]
    
    # Get last human message
    user_msg = next(msg for msg in reversed(messages) if isinstance(msg, HumanMessage))
    
    # Check if retriever is available
    global retriever
    if retriever is None:
        error_response = AIMessage(content="I'm sorry, but I need a YouTube video transcript to be loaded first before I can analyze video content. Please provide a YouTube URL and process it using the `process_youtube_video()` function.")
        return {
            "messages": messages + [error_response],
            "audio_file": None
        }
    
    # Get transcript context
    try:
        result = retriever.invoke(user_msg.content)
        context_docs = []
        for doc in result:
            metadata = doc.metadata
            content = doc.page_content
            # extract metadata variables
            minute = metadata.get("minute")
            start_timestamp = metadata.get("start_timestamp")
            # append context with metadata + content
            context_docs.append(f"Start Time: {start_timestamp}, Minute: {minute}\n{content}")

        context = "\n\n".join(context_docs)
        
        # Prepare conversation history for context
        conversation_history = "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages[:-1]  # Exclude current message
        )

        template_video_response = ChatPromptTemplate.from_messages([
            ("system", system_msg_video),
            ("human", f"Conversation History:\n{conversation_history}\n\n"
                     f"Current Question: {user_msg.content}\n\n"
                     f"Transcript Context:\n{context}")
        ])

        chain_video_response = template_video_response | llm
        response = chain_video_response.invoke({})
        
    except Exception as e:
        print(f"Error processing video query: {e}")
        response = AIMessage(content=f"I encountered an error while processing your video-related query: {str(e)}. Please make sure a video transcript has been properly loaded.")
    
    # Generate audio for the response
    audio_file = None
    try:
        audio_file = generate_audio_files(response.content)
        print(f"Generated audio for video Q&A: {audio_file}")
    except Exception as e:
        print(f"Error generating audio for video Q&A: {e}")
    
    return {
        "messages": messages + [response],
        "audio_file": audio_file
    }

# Build the workflow with 2 nodes
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
workflow = StateGraph(State)

# Add the two main nodes
workflow.add_node("route_query", route_query)
workflow.add_node("handle_general_query", handle_general_query)
workflow.add_node("handle_video_qa", handle_video_qa)

# Set entry point to router
workflow.set_entry_point("route_query")

# Add conditional edges from router to the two main nodes
workflow.add_conditional_edges(
    "route_query",
    lambda state: state.get("decision", "general_query"),
    {
        "general_query": "handle_general_query",
        "video_qa": "handle_video_qa"
    }
)

# Add edges from each node to END
workflow.add_edge("handle_general_query", END)
workflow.add_edge("handle_video_qa", END)

# Compile the app
app = workflow.compile(checkpointer=checkpointer)

# Generate and display the graph
try:
    graph_image = app.get_graph().draw_mermaid_png()
    display(Image(graph_image))
    # Save graph to file
    with open("workflow_graph.png", "wb") as f:
        f.write(graph_image)
    print("Workflow graph saved as 'workflow_graph.png'")
except Exception as e:
    print(f"Could not generate graph visualization: {e}")

print("YouTube Video Analysis Workflow is ready!")
print("\nThe workflow has two main capabilities:")
print("1. General Query Handler - For greetings, casual conversation, and general questions")
print("2. Video Q&A Handler - For video analysis, summarization, and content-based questions")
print("3. Audio files are automatically generated for all LLM responses")
