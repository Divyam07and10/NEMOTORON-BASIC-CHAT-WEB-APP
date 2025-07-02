import streamlit as st
from openai import OpenAI
import os
from rag.vector_store import get_context_relevant_to
import re

# Optional: load from .env
from dotenv import load_dotenv
load_dotenv()

# Secure API Key load
API_KEY = os.getenv("NVIDIA_API_KEY") or "your-fallback-key"

# Init OpenAI client with NVIDIA endpoint
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
)

# Streamlit App UI
st.set_page_config(page_title="Nemotron Chat", layout="centered")
st.title("NVIDIA Nemotron Chat App")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Simple function to detect potentially offensive language
def contains_offensive_language(text):
    # Basic regex for common offensive words (extend as needed)
    offensive_patterns = [
        r'\b(fuck|shit|damn|asshole|bitch|bastard)\b',
        r'\b(hell|crap|piss)\b'
    ]
    return any(re.search(pattern, text.lower()) for pattern in offensive_patterns)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check for offensive language
    if contains_offensive_language(prompt):
        response = "I apologize if you're frustrated. I'm here to assist with questions about the help guide content. Could you please rephrase your question?"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Retrieve relevant context from vector store
        context = get_context_relevant_to(prompt, k=3)

        # Check if context is empty
        if not context:
            response = "I apologize, but I can only assist with questions related to the help guide content. Could you please ask something related to that?"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # Construct system prompt with context
            system_prompt = (
                "You are a professional and courteous assistant. Respond in a concise, polite, and professional manner, "
                "even if the user is frustrated or uses inappropriate language. Use the following information to answer the "
                "user's question accurately:\n\n"
                f"{context}\n\n"
                "If the question isn't related to the provided information, politely state that you can only assist with questions "
                "about the help guide content. Do not use or repeat any offensive language in your response."
            )

            # Prepare messages for API call, including full history (limited to last 10 messages)
            messages = [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.messages[-10:]  # Include last 10 messages to maintain context

            # Placeholder for AI response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                # Streaming Nemotron response
                try:
                    completion = client.chat.completions.create(
                        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
                        messages=messages,
                        temperature=0.6,
                        top_p=0.95,
                        max_tokens=4096,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stream=True
                    )

                    # Display streamed chunks
                    for chunk in completion:
                        if chunk.choices[0].delta and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response)

                    # Save assistant response
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    response = f"I apologize, but an error occurred: {str(e)}. Please try again or ask a question related to the help guide."
                    response_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})