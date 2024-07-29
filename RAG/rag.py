import os
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tiktoken
import numpy
from scipy.spatial.distance import cosine
from langchain.load import dumps, load
from operator import itemgetter
import json

#### INDEXING ####
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODEL")
llm = ChatOpenAI(model = model)

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# print(type(docs[0]))

embedding = OpenAIEmbeddings(api_key = api_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_splitter.split_documents(docs)
# len(splits) 
# len(splits[0].page_content)

vectorstore = Chroma.from_documents(documents = splits, embedding = embedding)
retriever = vectorstore.as_retriever(search_kwargs = {"k" : 2})


template = """Answer the question based only on the following context:
{context}

Question : {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm
chain.invoke({"context" : docs, "question": "what is task decomposition"})


prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition")

## Indexing

question = "what kind of pet do I like?"
document = "My favorite pet is a cat!!"

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens =  len(encoding.encode(string))
    return num_tokens

encoding_name = tiktoken.encoding_name_for_model(model)
print(f"encoding name = {encoding_name}")
print(f"number of tokens = {num_tokens_from_string(question, encoding_name)}")

embedding = OpenAIEmbeddings(api_key = api_key)
query_result = embedding.embed_query(question)
document_result = embedding.embed_query(document)

def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

similarity = cosine_similarity(query_result, document_result)
print(f"The similarity betweeen the documents = {similarity}")

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm
chain.invoke({"context" : document, "question": question})

## MultiQuery

template = """You are an AI language model assistant and your task isto generate five different versions of the given user question to retrieve relevant documents from a vector databse.""" \
           """By generating multiple perspectives on the suer question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.""" \
           """Provide these alternative questions separated by newlines. Original question: {question}."""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    {"question" : RunnablePassthrough()}
    | prompt_perspectives 
    | llm
    | StrOutputParser()
    | (lambda x : x.split("\n"))
)

generate_queries.invoke("What is task decomposition?") 

def get_unique_union(documents: list[list]):
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [load(doc) for doc in unique_docs]

question = "what is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question" : question})
print(f"The length of docs = {len(docs)}")

template = """Answer the question based only on the following context:
{context}

Question : {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model = model, temperature = 0)

final_rag_chain = (
    {"context" : retrieval_chain, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke(question)

### RAG Fusion

template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x : x.split("\n"))
)

generate_queries.invoke(question)

# Now in RAG Fusion -> we have a reciprocal rank fucntion 

# def reciprocal_rank_fusion(results: list[list], k = 60):
#     fused_scores = {}
#     for docs in 


