import streamlit as st
import os
from ctransformers import AutoModelForCausalLM

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Function to load the LLaMA 2 model with caching
@st.cache_resource()
def ChatModel(temperature, top_p):
    model_path = 'models/llama-2-7b-chat.ggmlv3.q8_0.bin'  # Ensure this file exists locally
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return AutoModelForCausalLM.from_pretrained(
        model_path, 
        model_type='llama',
        temperature=temperature, 
        top_p=top_p
    )

# Sidebar for adjusting model parameters
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')

    st.subheader('Models and parameters')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=2.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

    chat_model = ChatModel(temperature, top_p)
    if not chat_model:
        st.stop()  # Stop execution if the model cannot be loaded

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    output = chat_model(f"prompt {string_dialogue} {prompt_input} Assistant: ")
    return output

# Handle user input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate and display assistant response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(st.session_state.messages[-1]["content"])
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    
    # Store the assistant's response
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
