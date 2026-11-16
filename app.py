import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from rag import answer_question, build_index, read_document, split_documents

load_dotenv()
st.set_page_config(page_title='Document Chat', page_icon='📚')
st.title('📚 Document Chat')
st.caption('Upload documents, ask questions, and inspect the supporting passages.')

for key, value in {'messages': [], 'store': None, 'signature': None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header('Your knowledge base')
    api_key = st.text_input('Google API key', value=os.getenv('GOOGLE_API_KEY', ''), type='password')
    chat_model = st.text_input('Chat model', os.getenv('CHAT_MODEL', 'gemini-2.5-flash-lite'))
    embedding_model = st.text_input('Embedding model', os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001'))
    files = st.file_uploader('PDF, TXT, or Markdown', type=['pdf', 'txt', 'md'], accept_multiple_files=True)
    st.info('Indexing sends document text to Google. Questions and retrieved passages are also sent to Google. Free-tier limits and data terms apply.')
    st.link_button('Get an API key', 'https://aistudio.google.com/apikey')
    if st.button('Clear chat'):
        st.session_state.messages = []
        st.rerun()

signature = hashlib.sha256((embedding_model + api_key).encode())
for file in files or []:
    signature.update(file.name.encode())
    signature.update(file.getvalue())
current_signature = signature.hexdigest()
if st.session_state.signature != current_signature:
    st.session_state.store = None
    st.session_state.messages = []
    st.session_state.signature = None

with st.sidebar:
    if st.button('Build knowledge base', disabled=not files or not api_key):
        try:
            with st.spinner('Reading and indexing documents…'):
                documents = []
                for file in files:
                    documents.extend(read_document(file.name, file.getvalue()))
                chunks = split_documents(documents)
                store = build_index(chunks, api_key, embedding_model)
            st.session_state.store = store
            st.session_state.signature = current_signature
            st.success(f'Ready: {len(chunks)} passages from {len(files)} files.')
        except Exception:
            st.error('Could not index these files. Check readable text, file limits, API key, model access, and quota. Try smaller files.')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('Supporting passages'):
                for source in message['sources']:
                    st.write(f"[{source['number']}] {source['metadata']}")
                    st.text(source['text'])

question = st.chat_input('Ask about your documents', disabled=st.session_state.store is None or not api_key)
if question:
    history = [{'role': m['role'], 'content': m['content']} for m in st.session_state.messages]
    with st.chat_message('user'):
        st.write(question)
    try:
        with st.spinner('Finding an answer…'):
            answer, sources = answer_question(question, st.session_state.store, api_key, chat_model, history)
        st.session_state.messages.extend([
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer, 'sources': sources}])
        st.rerun()
    except Exception:
        st.error('The model could not answer. Check your API key, model availability, internet connection, and free-tier quota, then retry.')
if st.session_state.store is None:
    st.info('Add your API key and files in the sidebar, then build the knowledge base.')
