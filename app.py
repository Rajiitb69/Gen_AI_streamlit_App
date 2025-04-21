import streamlit as st
import numpy as np
import os
import re
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
## Code #####
# Dummy credentials
PASSWORD = "Sherlocked"

system_prompt = """You are CodeGenie, an expert software engineer and coding tutor.
You are currently helping a user named {username}.

Your job is to help {username} with code suggestions, debugging, and explanations across programming languages like Python, Java, C++, JavaScript, SQL, etc.

Your reply style should be:
- Friendly and encouraging (start with phrases like "Great question!", "Sure!", or "Let's walk through it...")
- Clear, concise answers
- Relevant code blocks
- Helpful comments and explanations
- Address the user by name when appropriate
- Use the language asked by the user
- Keep extra text minimal, but don’t be robotic
"""

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'login'
if 'user_name' not in st.session_state:
    st.session_state.user_name = ''
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ''
if 'selected_screen' not in st.session_state:
    st.session_state.selected_screen = ''

# Login
def login_screen():
    st.title("🔐 Login")
    with st.form("login_form", clear_on_submit=True):
        user_name = st.text_input("Enter your Name:")
        secret_key = st.text_input("Enter Secret Key:", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if secret_key == PASSWORD:
                st.session_state.logged_in = True
                st.session_state.user_name = user_name
                st.session_state.step = 'greeting'
                st.rerun()  # To go straight to the main app
            elif user_name == '' or not user_name:
                st.error("Please enter your Name")
            elif not secret_key:
                st.error("Please enter Secret Key")
            else:
                st.error("Invalid Secret Key")

# Logout screen
def logout_sidebar(user_name):
    # Sidebar logout
    with st.sidebar:
        st.markdown("## 👤 User Panel")
        st.write(f"Logged in as: **{user_name}**")
        if st.button("🔒 Logout"):
            for key in ["step", "username", "user_name", "api_key", "selected_screen"]:
                st.session_state.pop(key, None)
            st.rerun()

def greeting_screen():
    user = st.session_state.user_name.title()
    logout_sidebar(user)
    # Stylish greeting
    st.markdown(f"""
        <div style='text-align: center;'>
            <h1 style='color:#4CAF50;'>👋 Hi {user}!</h1>
            <h4>Welcome to your <span style="color:#FF6F61;">Personal AI Assistant</span> 👨‍💻</h4>
            <p style='font-size:17px;'>Choose a tool you'd like to use. We'll ask for your <strong>GROQ API Key</strong> next.</p>
        </div>
    """, unsafe_allow_html=True)

    # Nice spacing
    st.markdown("---")
    # Select screen
    selected = st.selectbox("📂 Choose a tool or section:",
            [ "💻 Code Assistant", "🧮 Math Assistant", "🔎 RAG-based Chatbot", "📝 Text Summarization", "📺 Youtube/Website Content Summarization"])
    
    st.session_state.selected_screen = selected
    # Go button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Let's Go"):
            st.session_state.step = 'ask_api_key'
            st.rerun()

def api_key_screen():
    st.title("🔑 Enter Your GROQ API Key")
    groq_api_key = st.text_input("GROQ API Key", type="password")
    user = st.session_state.user_name.title()
    logout_sidebar(user)
    if st.button("Submit"):
        if not groq_api_key.strip():
            st.error("Please enter a valid API key.")
            return
        with st.spinner("🔄 Validating your GROQ API key..."):
            try:
                llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
                _ = llm.invoke([HumanMessage(content="Hello!")])
                st.session_state.groq_api_key = groq_api_key.strip()
                st.session_state.step = 'main'
                st.success("API key validated successfully! 🎉")
                st.rerun()
            except Exception as e:
                st.error("❌ Invalid GROQ API key")

# Streamlit UI                
def code_assistant_screen():
    
    groq_api_key = st.session_state.groq_api_key
    user_name = st.session_state.user_name.title()
    # Set sidebar width using CSS
    st.markdown("""<style>
                    section[data-testid="stSidebar"] {min-width: 250px; max-width: 250px; width: 250px;}
                </style>""", unsafe_allow_html=True)
    
    logout_sidebar(user_name)
            
    st.title("🤖 Your Coding Assistant")
    """
    It's a code assistant that provides you with answers to your queries. It helps users with code suggestions,
    debugging, and explanations across languages like Python, Java, C++, JavaScript, SQL, etc.
    """
    
    query = st.chat_input(placeholder="Write your query?")

    if "messages" not in st.session_state:
        st.session_state["messages"]=[
            {"role": "assistant", "content": f"Hi {user_name}, I'm a code assistant. How can I help you?"}
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])
    
    if user_name!='' and groq_api_key and query:
            
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)
        
        # Reconstruct chat history (excluding initial assistant greeting)
        chat_history = []
        for msg in st.session_state.messages[1:]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
    
        # Prompt template with username
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]).partial(username=user_name)
        
        llm3 = ChatGroq(model="llama-3.3-70b-versatile",
                       groq_api_key=groq_api_key,
                        temperature = 0.2,  # for randomness, low- concise & accurate output, high - diverse and creative output
                      max_tokens = 500,   # Short/long output responses (control length)
                        model_kwargs={
                                   "top_p" : 0.5,        # high - diverse and creative output
                                    })
        
        chain: Runnable = prompt_template | llm3
    
        with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=chain.invoke({"input": query,"chat_history": chat_history}, callbacks=[st_cb])
            final_answer = response.content if hasattr(response, "content") else str(response)
            st.write(final_answer)
            st.session_state.messages.append({'role': 'assistant', "content": final_answer})
            
    elif user_name!='' and groq_api_key and not query:
        st.warning("Please type a coding question to get started.")

def math_assistant_screen():
    st.subheader("🧮 Math Assistant")
    st.write("Your assistant is ready to help with code.")

def RAG_based_chatbot_screen():
    st.subheader("🔎 RAG-based Chatbot")
    st.write("Your assistant is ready to help with code.")

def text_summarization_screen():
    st.subheader("📝 Text Summarization")
    st.write("Your assistant is ready to help with code.")

def content_summarization_screen():
    st.subheader("📺 Youtube/Website Content Summarization")
    st.write("Your assistant is ready to help with code.")
    
# Dispatcher to selected screen
def main_router():
    selection = st.session_state.get("selected_screen", "💻 Code Assistant")
    if selection == "💻 Code Assistant":
        code_assistant_screen()
    elif selection == "🧮 Math Assistant":
        math_assistant_screen()
    elif selection == "🔎 RAG-based Chatbot":
        RAG_based_chatbot_screen()
    elif selection == "📝 Text Summarization":
        text_summarization_screen()
    elif selection == "📺 Youtube/Website Content Summarization":
        math_assistant_screen()
    else:
        st.warning("No screen selected.")

# App Flow Control
def run_app():
    if st.session_state.step == 'login':
        login_screen()
    elif st.session_state.step == 'greeting':
        greeting_screen()
    elif st.session_state.step == 'ask_api_key':
        api_key_screen()
    elif st.session_state.step == 'main':
        main_router()

# Start app
run_app()
