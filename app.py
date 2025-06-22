
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Streamlit page setup
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("🤖 Streamlit Chatbot using DialoGPT")

# Load model and tokenizer (cache to avoid reloading)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# Initialize chat history in session state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_inputs" not in st.session_state:
    st.session_state.past_inputs = []

# User input
user_input = st.text_input("You:", placeholder="Say something...", key="input")

if st.button("Send") and user_input:
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    # Generate a response
    output_ids = model.generate(
        new_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response
    response = tokenizer.decode(output_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Display bot's response
    st.write("bot:",response)

    bot_input_ids = torch.cat(
        [st.session_state.chat_history_ids, new_input_ids], dim=-1
    ) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate response
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
    )

    # Decode and display
    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Save interaction
    st.session_state.past_inputs
