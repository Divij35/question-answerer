from transformers import pipeline
import streamlit as st

answerer = pipeline("question-answering", model='bert-large-uncased-whole-word-masking-finetuned-squad')

def get_context():
    with open("info.txt") as file:
        context = file.read()
    return context

context = get_context()

st.title("Question Answering on solar system")

st.write("Ask a question based on the content of the file.")

question = st.text_input("Enter your question:")

if question:
    result = answerer(question=question, context=context)
    answer = result['answer']
    st.write(f"**Answer:** {answer}")

st.divider()
st.warning("Please have a look at the context if desired output is not obtained", icon="⚠️")

st.write(context)