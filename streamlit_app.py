import streamlit as st, os, requests
st.set_page_config(layout='wide')
st.title('RAG Chatbot - Demo')

API = st.text_input('API URL', value='http://localhost:8000')
question = st.text_area('Ask:', height=120)
top_k = st.slider('Top-K', 1, 10, 5)
if st.button('Ask'):
    resp = requests.post(API + '/ask', json={'question': question, 'top_k': top_k})
    if resp.status_code == 200:
        data = resp.json()
        st.subheader('Generative Answer')
        st.write(data.get('generative'))
        st.subheader('Extractive Answer')
        st.json(data.get('extractive'))
        st.subheader('Hits')
        for h in data.get('hits', []):
            st.write(h.get('text')[:400] + '...')
    else:
        st.error(resp.text)
