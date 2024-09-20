from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0,
            tokens_per_chunk=256,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

vector_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def pretty_docs(docs):
    smz_docs = ""
    for i, doc in enumerate(docs):
        smz_docs += f"  \n  \nDocument {i+1}:"
        smz_docs += f"  \nPage: {doc.metadata.get('page')} and Source: {doc.metadata.get('source')}"
        smz_docs += f"  \nContent: {doc.page_content[:100]}..."
    return smz_docs


def text_extract(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
        )
    split_texts = loader.load_and_split(text_splitter)
    print(f"Split document into {len(split_texts)} chunks")
    return split_texts


def token_text_split(split_texts):
    tokens = token_splitter.split_documents(split_texts)
    print(f"Split document into {len(tokens)} chunks")
    return tokens


def get_embeddings(split_tokens, c_name):
    vector_store = Chroma.from_documents(documents=split_tokens,
                                         embedding=vector_embedding,
                                         collection_name=c_name,
                                         persist_directory="vector_directory")
    print(f"Created vector store with {vector_store._collection.count()} embeddings")
    return vector_store


def process_document(folder_path, collection="sample-set"):
    split_texts = text_extract(folder_path)
    split_tokens = token_text_split(split_texts)
    vector_store = get_embeddings(split_tokens, collection)
    return vector_store
