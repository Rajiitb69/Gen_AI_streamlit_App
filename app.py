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
PASSWORD = "1234"

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
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ''

# Login
def login():
    st.title("🔐 Login")
    with st.form("login_form", clear_on_submit=True):
        user_name = st.text_input("Enter your Name:")
        secret_key = st.text_input("Enter Secret Key:", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if secret_key == PASSWORD:
                st.session_state.logged_in = True
                st.session_state.user_name = user_name
                st.success("Login successful!")
                st.rerun()  # To go straight to the main app
            elif user_name == '' or not user_name:
                st.error("Please enter your Name")
            elif not secret_key:
                st.error("Please enter Secret Key")
            else:
                st.error("Invalid Secret Key")
# Streamlit UI                
def main_app(user_name):
    st.title("🤖 Your Coding Assistant")
    """
    It's a code assistant that provides you with answers to your queries. It helps users with code suggestions,
    debugging, and explanations across languages like Python, Java, C++, JavaScript, SQL, etc.
    """
    
    ## Sidebar for settings
    st.sidebar.title("INPUTS")
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key:",type="password")
    
    query = st.chat_input(placeholder="Write your query?")

    if "messages" not in st.session_state:
        st.session_state["messages"]=[
            {"role": "assistant", "content": f"Hi {user_name.title()}, I'm a code assistant. How can I help you?"}
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
    elif not groq_api_key:
        st.info("👈 Please enter Groq API key in the sidebar to continue.")


# App logic
if not st.session_state.logged_in:
    login()
else:
    main_app(st.session_state.user_name)

