import streamlit as st
from backend import process_youtube_video , app
from streamlit_mic_recorder import speech_to_text
from langchain_core.messages import HumanMessage , AIMessage
import os
import uuid

# Initialize session state
if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

if "video_loaded" not in st.session_state:
    st.session_state['video_loaded'] = False

if "current_video_url" not in st.session_state:
    st.session_state['current_video_url'] = ""

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

if "waiting_for_response" not in st.session_state:
    st.session_state['waiting_for_response'] = False

# --- Sidebar for YouTube URL + Language Selection ---
st.sidebar.header("🎥 YouTube Video Settings")
youtube_url = st.sidebar.text_input(
    "Enter YouTube URL", 
    value=st.session_state.get('current_video_url', ''),
    placeholder="https://www.youtube.com/watch?v=..."
)
language = st.sidebar.selectbox(
    "Select Transcript Language", 
    ["en", "hi", "fr", "es", "de"],
    help="Choose the language for video transcript extraction"
)
load_button = st.sidebar.button("🔄 Load Transcript", type="primary")

# Handle video loading
if load_button and youtube_url:
    if youtube_url != st.session_state.get('current_video_url', ''):
        # Clear chat history when loading new video
        st.session_state['message_history'] = []
        # Generate new thread ID for new video
        st.session_state['thread_id'] = str(uuid.uuid4())
    
    with st.spinner("Loading transcript and building vector database..."):
        try:
            process_youtube_video(youtube_url, language)
            st.session_state['video_loaded'] = True
            st.session_state['current_video_url'] = youtube_url
            st.sidebar.success("✅ Transcript loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading transcript: {str(e)}")
            st.session_state['video_loaded'] = False

# Show current video status
if st.session_state['video_loaded']:
    st.sidebar.info(f"✅ Video loaded: {st.session_state['current_video_url'][:50]}...")
else:
    st.sidebar.warning("⚠️ No video loaded. Load a video to enable video-related Q&A.")

# --- Audio Settings ---
st.sidebar.markdown("---")
st.sidebar.header("🔊 Audio Settings")
auto_play_audio = st.sidebar.checkbox(
    "Auto-play response audio", 
    value=True,
    help="Automatically play audio when AI responds"
)

# --- Main Chat Section ---
st.title("🤖 YouTube Conversational AI")
st.markdown("Ask me anything! I can help with general questions or analyze YouTube videos.")

# Show capabilities info
with st.expander("ℹ️ What can I do?", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🗨️ General Conversation:**
        - Answer general questions
        - Casual conversation
        - Explain my capabilities
        - Help with various topics
        """)
    with col2:
        st.markdown("""
        **🎥 Video Analysis:**
        - Summarize YouTube videos
        - Answer specific questions about content
        - Find information with timestamps
        - Analyze video topics in detail
        """)

# Create a container for the chat messages
chat_container = st.container()

# Display chat history
with chat_container:
    for i, message in enumerate(st.session_state['message_history']):
        with st.chat_message(message['role']):
            st.write(message['content'])
            
            # Add audio player for assistant messages if audio file exists
            if message['role'] == 'assistant' and 'audio_file' in message and message['audio_file']:
                if os.path.exists(message['audio_file']):
                    st.audio(message['audio_file'], format='audio/mp3')
                else:
                    st.caption("🔇 Audio file not found")

# Input section at the bottom
st.markdown("---")

# Create columns for different input methods
input_col1, input_col2 = st.columns([2, 8])

with input_col1:
    st.markdown("**🎤 Voice Input:**")
    # Speech to text input
    speech_text = speech_to_text(
        language="en", 
        start_prompt="🎤 Record", 
        stop_prompt="⏹️ Stop", 
        key="speech_input",
        just_once=True,
        use_container_width=True
    )

with input_col2:
    st.markdown("**⌨️ Text Input:**")
    # Text input
    user_input = st.chat_input(
        "Type your message here..." if not st.session_state['waiting_for_response'] else "Please wait for the current response...",
        disabled=st.session_state['waiting_for_response']
    )

# Process input (prioritize text input over speech)
final_input = None

if user_input and not st.session_state['waiting_for_response']:
    final_input = user_input
elif speech_text and not st.session_state['waiting_for_response']:
    final_input = speech_text
    st.success(f"🎤 Voice input captured: \"{speech_text}\"")

if final_input:
    # Set waiting state
    st.session_state['waiting_for_response'] = True
    
    # Add user message to history
    st.session_state['message_history'].append({
        "role": "user", 
        "content": final_input,
        "audio_file": None
    })
    
    # Display user message immediately
    with chat_container:
        with st.chat_message("user"):
            st.write(final_input)

    # Show assistant "thinking" placeholder
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            audio_placeholder = st.empty()
            message_placeholder.write("🤔 Processing your request...")

            try:
                # Use the app directly with proper config
                config = {"configurable": {"thread_id": st.session_state['thread_id']}}
                
                response = app.invoke(
                    {"messages": [HumanMessage(content=final_input)]}, 
                    config=config
                )
                
                # Extract response and audio file from the result
                ai_response = response['messages'][-1].content
                audio_file = response.get('audio_file')

                # Update message placeholder with final response
                message_placeholder.write(ai_response)

                # Add audio player if audio file was generated
                if audio_file and os.path.exists(audio_file):
                    audio_placeholder.audio(audio_file, format='audio/mp3')
                    if auto_play_audio:
                        # Note: Streamlit doesn't support auto-play due to browser restrictions
                        st.caption("🔊 Audio generated! Click play button above to listen.")
                else:
                    audio_placeholder.caption("🔇 Audio generation failed or not available")

                # Save assistant response to history
                st.session_state['message_history'].append({
                    "role": "assistant", 
                    "content": ai_response,
                    "audio_file": audio_file
                })

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                message_placeholder.write(error_msg)
                st.session_state['message_history'].append({
                    "role": "assistant", 
                    "content": error_msg,
                    "audio_file": None
                })
            
            finally:
                # Reset waiting state
                st.session_state['waiting_for_response'] = False
                # Rerun to update the input field state
                st.rerun()

# --- Sidebar Controls ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Controls")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state['message_history'] = []
        st.session_state['waiting_for_response'] = False
        st.rerun()

with col2:
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state['message_history'] = []
        st.session_state['video_loaded'] = False
        st.session_state['current_video_url'] = ""
        st.session_state['waiting_for_response'] = False
        st.session_state['thread_id'] = str(uuid.uuid4())
        st.rerun()

# --- Status and Statistics ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Session Info")
st.sidebar.metric("Messages", len(st.session_state['message_history']))
st.sidebar.metric("Thread ID", st.session_state['thread_id'][:8] + "...")

if st.session_state['video_loaded']:
    st.sidebar.success("🎥 Video Analysis Ready")
else:
    st.sidebar.info("💬 General Chat Mode")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
🤖 AI Assistant with Voice & Video Analysis<br>
Powered by LangGraph & Groq
</div>
""", unsafe_allow_html=True)

# Display some sample queries for new users
if len(st.session_state['message_history']) == 0:
    st.markdown("---")
    st.markdown("### 💡 Try asking me:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **General Questions:**
        - "Hello! What can you do?"
        - "How does AI work?"
        - "Tell me about yourself"
        """)
    
    with col2:
        if st.session_state['video_loaded']:
            st.markdown("""
            **Video Questions:**
            - "Summarize this video"
            - "What are the main points?"
            - "Tell me about the 5-minute mark"
            """)
        else:
            st.markdown("""
            **Video Questions:**
            - *Load a video first to ask video-specific questions*
            - *Then try: "Summarize this video"*
            - *Or: "What are the main points?"*
            """)

# Auto-scroll to bottom (JavaScript injection)
st.markdown(
    """
    <script>
    window.parent.document.querySelector('section.main').scrollTo(0, window.parent.document.querySelector('section.main').scrollHeight);
    </script>
    """,
    unsafe_allow_html=True
)