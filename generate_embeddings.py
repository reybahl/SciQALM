from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

uri = "mongodb+srv://bahlreyansh:OLRgjshrXD3GN0MW@cluster0.ivvpa83.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.sample_mflix
collection = db.movies

docs = collection.find({"plot" : {"$exists" : True}}).limit(500) # find 500 docs in the dataset

for doc in docs:
    doc["plot_embedding_hf"] = model.encode(doc["plot"]).tolist()
    print(doc["_id"])
    collection.replace_one({"_id" : doc["_id"]}, doc)

