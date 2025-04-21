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

code_assistant_prompt = """You are CodeGenie, an expert software engineer and coding tutor.
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
code_assistant_title = "🤖 Your Coding Assistant"
code_assistant_header = """
    It's a code assistant that provides you with answers to your queries. It helps users with code suggestions,
    debugging, and explanations across languages like Python, Java, C++, JavaScript, SQL, etc.
    """
math_assistant_prompt = """
You are an expert mathematics tutor helping a user named {username}.
Your job is to solve mathematical questions of all kinds, including arithmetic, algebra, geometry, calculus, statistics, linear algebra, and word problems.
Please follow these guidelines:
1. Break the problem down into clear, logical steps.
2. Show all work for calculations and justify any rules or theorems used.
3. If the problem is word-based, first extract the relevant information and formulate equations.
4. If multiple approaches exist, briefly mention them, and then choose the most efficient one.
5. If the final answer is numerical, round it to a reasonable number of decimal places if needed.
6. Use LaTeX formatting to display math expressions cleanly if supported.
Your reply style should be friendly and encouraging (start with phrases like "Great question!", "Sure!", or "Let's walk through it...")
Let’s begin solving the problem.
"""
math_assistant_title = "🤖 Your Math Assistant"
math_assistant_header = """
    Welcome to your personal **Math Assistant**!
    Just type your question and let the assistant guide you through the solution! 💡
    """
text_summarization_prompt = """
You are an expert language assistant helping a user named {username}.
Your task is to read and summarize any text provided by the user — whether it's an article, email, report, research paper, or story.

Please follow these guidelines:
1. Understand the main idea and supporting details of the text.
2. Summarize the content clearly and concisely, keeping the most important points.
3. Maintain the original tone and intent (informative, persuasive, formal, etc.).
4. If the text is technical or academic, retain key terminology but simplify explanations where helpful.
5. Avoid including unnecessary examples or repetition from the original.
6. Offer summaries in bullet points if appropriate, or in a short paragraph for general content.
7. Keep your tone helpful and friendly (start with phrases like "Sure!", "Here's a quick summary:", etc.).
Let’s get started with the summarization.
"""
text_summarization_title = "🤖 Your Text Summarizer"
text_summarization_header = """
    Welcome to your personal **Text Summarizer**!
    Summarize articles, emails, reports, or any text in seconds. 
    Just paste the content, and get a clear, concise summary! 💡
    """

def get_prompt(tool, user_name):
    if tool == "💻 Code Assistant":
        title = code_assistant_title
        header = code_assistant_header
        assistant_content = f"Hi {user_name}, I'm a code assistant. How can I help you?"
        system_prompt = code_assistant_prompt
    elif tool == "🧮 Math Assistant":
        title = math_assistant_title
        header = math_assistant_header
        assistant_content = f"Hi {user_name}, I'm a math assistant. How can I help you?"
        system_prompt = math_assistant_prompt
    elif tool == "📝 Text Summarization":
        title = text_summarization_title
        header = text_summarization_header
        assistant_content = f"Hi {user_name}, I'm a Text Summarizer. How can I help you?"
        system_prompt = text_summarization_prompt
    output_dict = {"title":title, "header":header, "assistant_content":assistant_content, "system_prompt":system_prompt}
    return output_dict

def get_layout(tool):
    groq_api_key = st.session_state.groq_api_key
    user_name = st.session_state.user_name.title()
    logout_sidebar(user_name)
    output_dict = get_prompt(tool, user_name)
    st.title(output_dict['title'])
    output_dict['header']
    
    query = st.chat_input(placeholder="Write your query?")
    
    # Toggle to expand/collapse all previous responses
    if "expand_all" not in st.session_state:
        st.session_state.expand_all = True
    
    if st.button("🔄 Toggle Expand/Collapse Previous Responses"):
        st.session_state.expand_all = not st.session_state.expand_all

    if "messages" not in st.session_state:
        st.session_state["messages"]=[
            {"role": "assistant", "content": output_dict['assistant_content']}
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
            ("system", output_dict['system_prompt']),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]).partial(username=user_name)
        
        llm3 = ChatGroq(model="llama-3.3-70b-versatile",
                       groq_api_key=groq_api_key,
                        temperature = 0.2,  # for randomness, low- concise & accurate output, high - diverse and creative output
                      max_tokens = 600,   # Short/long output responses (control length)
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
        
        for i, msg in enumerate(st.session_state.messages[:-1]):  # exclude latest
            if msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    with st.expander(f"🧠 Response #{i//2 + 1}", expanded=st.session_state.expand_all):
                        st.markdown(msg["content"])
            
    elif user_name!='' and groq_api_key and not query:
        st.warning("Please type a query to get started.")

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
    # Set sidebar width using CSS
    st.markdown("""<style>
                    section[data-testid="stSidebar"] {min-width: 250px; max-width: 250px; width: 250px;}
                </style>""", unsafe_allow_html=True)
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
            ["💻 Code Assistant", "🧮 Math Assistant", "📝 Text Summarization", "🔎 RAG-based Chatbot", "📺 Youtube/Website Content Summarization"])
    
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
def code_assistant_screen(selection):
    get_layout(selection)

def math_assistant_screen(selection):
    get_layout(selection)

def RAG_based_chatbot_screen(selection):
    groq_api_key = st.session_state.groq_api_key
    user_name = st.session_state.user_name.title()
    logout_sidebar(user_name)
    st.title("🤖 Your RAG Based Chatbot")

def text_summarization_screen(selection):
    get_layout(selection)

def content_summarization_screen(selection):
    st.subheader("📺 Youtube/Website Content Summarization")
    groq_api_key = st.session_state.groq_api_key
    user_name = st.session_state.user_name.title()
    logout_sidebar(user_name)
    st.title("🤖 Your Content Summarization Assistant")
    
# Dispatcher to selected screen
def main_router():
    selection = st.session_state.get("selected_screen", "💻 Code Assistant")
    if selection == "💻 Code Assistant":
        code_assistant_screen(selection)
    elif selection == "🧮 Math Assistant":
        math_assistant_screen(selection)
    elif selection == "🔎 RAG-based Chatbot":
        RAG_based_chatbot_screen(selection)
    elif selection == "📝 Text Summarization":
        text_summarization_screen(selection)
    elif selection == "📺 Youtube/Website Content Summarization":
        math_assistant_screen(selection)
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
