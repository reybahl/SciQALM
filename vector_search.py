from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

uri = "mongodb+srv://bahlreyansh:OLRgjshrXD3GN0MW@cluster0.ivvpa83.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client.sample_mflix
collection = db.movies

def generate_context(query):
    # query = "who does Space hero Daffy battle for control of planet x"

    results = collection.aggregate([
        {
            "$vectorSearch" : {
                "queryVector" : model.encode(query).tolist(),
                "path" : "plot_embedding_hf",
                "numCandidates" : 2000,
                "limit" : 10,
                "index" : "PlotSemanticSearch"
            }
        }
    ])

    context = ""

    for document in results:
        context += f"Movie Name: {document['title']}\nMovie Plot: {document['plot']}\n\n"

    return context