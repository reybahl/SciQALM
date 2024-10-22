from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

uri = open("../mongodburl.txt").read()
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

docs = collection.find({"plot" : {"$exists" : True}}).limit(20000) # find 2000 docs in the dataset

for doc in tqdm(docs):
    doc["plot_embedding_hf"] = model.encode(doc["plot"]).tolist()
    collection.replace_one({"_id" : doc["_id"]}, doc)
