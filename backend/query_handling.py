from backend.model_integration import llm
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import List

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))


def multi_query_retriever(retriever):
    print("Starting Multi query retrieval")
    output_parser = LineListOutputParser()

    prompt_template = """
        You are a helpful and creative assistant with a goal to assist users.
        When the user asks a question, your task is to generate 5 semantically
        similar questions to it, to help them find the information they need.
        Make sure that the questions are diverse and cover different aspects
        of the topic. Suggest only short queries without compound sentences.
        Output one question per line. Do not number the questions.
        User question: {question}

        Similar questions:
    """

    q_prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    print("Setting up llm chain")

    model = llm()

    llm_chain = q_prompt | model | output_parser

    print("Setting up MQR chain")

    multi_retriever = MultiQueryRetriever(
        include_original=True,
        retriever=retriever,
        llm_chain=llm_chain,
        parser_key="lines"
    )

    return multi_retriever


def rag_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert helpful assistant.
        Answer the following question based on the provided context. For any question,
        if you cannot answer the question from the context, just say "I don't know".
        Keep the answers concise for the most part.
    
        Context: {context}
    
        Question: {input}
    
        Answer:
        """
    )

    print("Setting up stuff doc chain")

    model = llm()

    document_chain = create_stuff_documents_chain(model, prompt)

    print("Getting MQR chain and setting up compression retriever")

    multi_retriever = multi_query_retriever(retriever)

    compressor = CrossEncoderReranker(model=cross_encoder, top_n=6)  # Set the number of doc for context
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multi_retriever
    )

    print("setting up final rag chain")
    rag_rerank_chain = create_retrieval_chain(compression_retriever, document_chain)
    return rag_rerank_chain


def get_llm_response(query, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    rag = rag_chain(retriever)
    print("Getting response")
    response = rag.invoke({"input": query})
    print("response generated")
    return response["answer"], response["context"]
