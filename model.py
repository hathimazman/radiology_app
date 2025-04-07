import os
from sentence_transformers import SentenceTransformer

# Set the model name and output directory
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LOCAL_MODEL_PATH = 'local_model'

# Create the output directory if it doesn't exist
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

# Download the model
print(f"Downloading model {MODEL_NAME} to {LOCAL_MODEL_PATH}...")
model = SentenceTransformer(MODEL_NAME, cache_folder=LOCAL_MODEL_PATH)

# Save the model to the local directory
model.save(LOCAL_MODEL_PATH)
print(f"Model successfully downloaded and saved to {LOCAL_MODEL_PATH}")

# Verify the model can be loaded back
print("Verifying model can be loaded from local path...")
loaded_model = SentenceTransformer(LOCAL_MODEL_PATH)
print("Model successfully loaded from local path")