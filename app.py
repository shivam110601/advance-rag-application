import streamlit as st
import os
from backend.document_processing import process_document, pretty_docs
from backend.query_handling import get_llm_response

st.subheader("Document QA Bot")

# Initialize variables
if 'chroma_db' not in st.session_state:
    st.session_state['chroma_db'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []


if st.session_state['chroma_db'] is None:
    st.info("Please upload a document to start querying.")

    # Set the directory where PDFs will be saved
    UPLOAD_DIRECTORY = "uploaded_pdf/"

    # Create the directory if it doesn't exist
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)

    uploaded_files = st.sidebar.file_uploader("Upload a PDF document", type="pdf", accept_multiple_files=True)

    # Process the document once it's uploaded
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)

            # Save each uploaded PDF in the directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Show success message for each file
            st.sidebar.success(f"Uploaded file: {uploaded_file.name}")

        # Process the uploaded document
        with st.sidebar.status('Processing document'):
            st.session_state['chroma_db'] = process_document(UPLOAD_DIRECTORY, "user-doc")

if st.session_state['chroma_db']:
    # Show messages in Chat box
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # For LLM response
            if message["role"] == "assistant":
                st.markdown(message["content"])
                with st.expander("Source Documents"):
                    st.markdown(pretty_docs(message["context"]))
            # For User query
            else:
                st.markdown(message["content"])

    # For current User query
    if query := st.chat_input("Ask a question related to the document"):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            response, source_docs = get_llm_response(query, st.session_state['chroma_db'])
            st.session_state.messages.append({"role": "assistant", "content": response, "context":source_docs})
            st.markdown(response)
            with st.expander("Source Documents"):
                st.markdown(pretty_docs(source_docs))
