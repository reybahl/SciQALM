from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

uri = open("mongodburl.txt").read()

client = MongoClient(uri, server_api=ServerApi('1'))
db = client.arxiv
collection = db.embedded_abstracts

def generate_context(query):
    results = collection.aggregate([
        {
            "$vectorSearch" : {
                "queryVector" : model.encode(query).tolist(),
                "path" : "abstract_encoded",
                "numCandidates" : 2000,
                "limit" : 10,
                "index" : "default"
            }
        }
    ])

    context = ""

    for document in results:
        context += f"Paper Title: {document['title']}\nAuthors: {document['authors']}\nAbstract: {document['abstract']}\n\n"

    return context