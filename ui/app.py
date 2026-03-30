import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.title("📚 RAG Chatbot")

query = st.text_input("Ask your question")

if st.button("Ask"):
    if query:
        response = requests.post(API_URL, json={"question": query})

        if response.status_code == 200:
            answer = response.json()["answer"]
            st.write("### Answer:")
            st.write(answer)
        else:
            st.error("Error from API")