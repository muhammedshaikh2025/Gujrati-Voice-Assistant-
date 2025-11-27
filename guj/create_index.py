import pandas as pd
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import numpy as np
import json

# 1. Load embedding model (Downloaded once)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Load CSV
df = pd.read_csv('/home/pi/Downloads/gujrati/guj/qa_gujarati.csv')
questions = df['question'].tolist()

# 3. Convert questions to vectors
print("Converting questions to vectors...")
question_vectors = model.encode(questions)

# 4. Annoy index create
dimension = question_vectors.shape[1]  # typically 384
index = AnnoyIndex(dimension, 'angular')

# 5. Add vectors into Annoy Index
print("Building Annoy index...")
for i, vec in enumerate(question_vectors):
    index.add_item(i, vec)

index.build(10)  # number of trees
index.save('my_annoy_index.ann')

print("Index saved: my_annoy_index.ann")

# 6. Save question mapping (very important)
mapping = {"index_to_question": questions}
with open("question_map.json", "w") as f:
    json.dump(mapping, f)

print("Mapping file saved: question_map.json")

print("Setup complete!")
