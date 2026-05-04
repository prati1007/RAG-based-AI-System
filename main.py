from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("vector_store", embeddings)

query = "Best model for low latency edge deployment?"
docs = db.similarity_search(query)

llm = OpenAI()
response = llm(f"Answer based on: {docs}\nQuestion: {query}")

print(response)
