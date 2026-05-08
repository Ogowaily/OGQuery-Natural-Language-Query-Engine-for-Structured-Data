
from  ogquery import OGQuery
#APi method test
engine = OGQuery(config={
    "data_dir": "./data",
    "api_keys": {
        "groq": ""
    },
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 5
})

engine.serve(host="127.0.0.1", port=8000)
#Direct method test
from ogquery import OGQuery
 
engine = OGQuery(config={
    "data_dir": "./data",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 5,
    "api_keys": {
        "groq": " "
    }
})

print("Uploading dataset...")

dataset_id = engine.upload(r" ", name="test")

print("Dataset ID:", dataset_id)

print("Running query...")

result = engine.query(dataset_id, " ")

print(result)


