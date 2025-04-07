from sentence_transformers import SentenceTransformer

# Download the model to a specific directory
local_model_path = "./local_model"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=local_model_path)

# This will download all necessary files to the local_model directory
print(f"Model downloaded to {local_model_path}")