import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
import uuid
from langchain_core.runnables import RunnableSequence

from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Load and preprocess Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Create a summary of the dataset
total_passengers = len(df)
survived = df['Survived'].sum()
survival_rate = (survived / total_passengers) * 100
avg_fare = df['Fare'].mean()
avg_age = df['Age'].dropna().mean()

dataset_summary = f"""
The Titanic dataset contains information about {total_passengers} passengers:
- {survived} passengers survived ({survival_rate:.1f}% survival rate)
- Average ticket fare was ${avg_fare:.2f}
- Average passenger age was {avg_age:.1f} years
- Classes: First, Second, and Third class
- Ports of Embarkation: S (Southampton), C (Cherbourg), Q (Queenstown)
"""

# Initialize OpenRouter LLM
llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=1000,
    temperature=0.7,
    streaming=True,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Titanic Chatbot"
    }
)

# Create the conversation prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a helpful assistant knowledgeable about the Titanic disaster and its passenger data.
        Use the following dataset summary to help answer questions:
        {dataset_summary}
        
        If the question requires specific statistics, use the provided summary.
        If the question is about the Titanic in general, use your knowledge to provide accurate historical context.
        
        Previous conversation context is also provided to help maintain coherent dialogue.
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Session Management
# Session Management
def init_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now()
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            return_messages=True,
            output_key='output',
            input_key='input'
        )
    if 'chain' not in st.session_state:
        # Create a RunnableSequence by combining the prompt and the llm
        st.session_state.chain = RunnableSequence(prompt, llm)
    if 'messages' not in st.session_state:
        st.session_state.messages = []


def check_session_timeout():
    if 'start_time' in st.session_state:
        if datetime.now() - st.session_state.start_time > timedelta(minutes=30):
            # Reset session after 30 minutes
            for key in ['session_id', 'start_time', 'memory', 'chain', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            init_session()
            return True
    return False

# Initialize session
init_session()

# Streamlit UI
st.title("🚢 Titanic Dataset Chatbot")
st.markdown("""
This chatbot can answer questions about the Titanic disaster and its passengers.
You can ask about survival statistics, passenger demographics, or historical context.
""")

# Check for session timeout
if check_session_timeout():
    st.warning("Your session has expired. Starting a new session.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask a question about the Titanic..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.chat_message("assistant"):
     with st.spinner("Thinking..."):
        try:
            # Update conversation context
            response = st.session_state.chain.invoke({
                "input": user_input,
                "dataset_summary": dataset_summary,
                "history": st.session_state.memory.load_memory_variables({})["history"]
            })
            # Extract and display only the content
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            
            # Update session time
            st.session_state.start_time = datetime.now()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.markdown("I apologize, but I encountered an error. Please try asking your question again.")

# Add dataset statistics in the sidebar
with st.sidebar:
    st.header("Dataset Statistics")
    st.markdown(dataset_summary)
    
    # Add session information
    st.header("Session Information")
    st.markdown(f"""
    - Session ID: {st.session_state.session_id[:8]}...
    - Session Start: {st.session_state.start_time.strftime('%Y-%m-%d %H:%M:%S')}
    - Messages in History: {len(st.session_state.messages)}
    """)
    
    # Add some example questions
    st.header("Example Questions")
    st.markdown("""
    - What was the survival rate on the Titanic?
    - How much did tickets cost on average?
    - What were the different passenger classes?
    - Which port did most passengers embark from?
    - What was the average age of passengers?
    - Tell me about the passengers in first class.
    - Were children more likely to survive?
    """)
