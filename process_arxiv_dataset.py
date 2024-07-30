import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
df = pd.read_json("hf://datasets/gfissore/arxiv-abstracts-2021/arxiv-abstracts.jsonl.gz", lines=True, chunksize=500)
uri = "mongodb+srv://bahlreyansh:OLRgjshrXD3GN0MW@cluster0.ivvpa83.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))
db = client.arxiv
collection = db.embedded_abstracts

chunk_count = 0

for chunk in df:
    df_dict = chunk.to_dict('list')
    columns = list(chunk.columns)
    for i in range(len(df_dict["id"])):
        data = {}
        for x in columns:
            data[x] = df_dict[x][i]
        data["_id"] = str(data["id"])
        data["abstract_encoded"] = model.encode(data["abstract"]).tolist()
        result = collection.insert_one(data)
    chunk_count += 1
    print(chunk_count)