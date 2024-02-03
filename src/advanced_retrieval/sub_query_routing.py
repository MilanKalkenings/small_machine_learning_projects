import os
from config import EMBED_NAME
from llama_index.embeddings import HuggingFaceEmbedding
from config import OPENAI_API_KEY
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.schema import Document


def openai_setup():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    llm = OpenAI()
    embed_model = HuggingFaceEmbedding(model_name=EMBED_NAME)
    return llm, embed_model
_, _ = openai_setup()





print("~ Sub-Question Retrieval, routing over two query engines ~")
attributes_ben = ["author Tolkien", "author Haraway"]
nodes_ben = []
for i in range(len(attributes_ben)):
    nodes_ben.append(Document(text=attributes_ben[i], id=f"x{i}"))
vqe_ben = VectorStoreIndex.from_documents(nodes_ben).as_query_engine()
print("'bens_interests_query_engine' retrieves over docs:")
for i, attribute in enumerate(attributes_ben):
    print(f"ben{i+1}: '{attribute}'")

attributes_john = ["author Martin", "author Tolkien", "blogposts about cats"]
nodes_john = []
for i in range(len(attributes_john)):
    nodes_john.append(Document(text=attributes_john[i], id=f"y{i}"))
vqe_john = VectorStoreIndex.from_documents(nodes_john).as_query_engine(similarity_top_k=5)
print("\n'johns_interests_query_engine' retrieves over docs:")
for i, attribute in enumerate(attributes_john):
    print(f"john{i+1}: '{attribute}'")


from llama_index.question_gen import OpenAIQuestionGenerator
query = "Which authors are liked by Ben and by John? No further information"
print("\n\nQuery:", query)
tools = [
    QueryEngineTool(
        query_engine=vqe_john,
        metadata=ToolMetadata(name="johns_interests_query_engine", description="what John likes to read")),
    QueryEngineTool(
        query_engine=vqe_ben,
        metadata=ToolMetadata(name="bens_interests_query_engine", description="what Ben likes to read"))]
sub = OpenAIQuestionGenerator.from_defaults()
qgen = SubQuestionQueryEngine.from_defaults(query_engine_tools=tools, verbose=True)
response = qgen.query(query)
print("Final response:", response)
"""
source_nodes = response.source_nodes
for source_node in source_nodes:
    print(source_node.text, "\n")
"""