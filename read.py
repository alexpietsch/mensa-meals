import sqlite3
import sys
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

instruction = "Represent the Food meal name:"

# model = SentenceTransformer('all-mpnet-base-v2', device="cuda" if torch.cuda.is_available() else "cpu")
# read model name from "model.txt"
model_name = open("model.txt", "r").read()
model = SentenceTransformer(model_name)
# model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

def encode_embedding(query):
    eingabe_embedding = model.encode([query]).astype('float32')
    return eingabe_embedding

def find_similar(query, k=5):
    eingabe_embedding = encode_embedding(query)
    # Calc query embedding
    # eingabe_embedding = model.encode([query]).astype('float32')
    print(eingabe_embedding)

    # FAISS-Index
    index = faiss.read_index("gerichte.index")
    ids = np.load("gerichte_ids.npy")

    # Search similar
    _, ind = index.search(eingabe_embedding, k)
    ind = ind[0]

    # Meal name
    conn = sqlite3.connect("gerichte.db")
    c = conn.cursor()
    results = []
    for i in ind:
        c.execute("SELECT name FROM gerichte WHERE id = ?", (int(ids[i]),))
        results.append(c.fetchone()[0])
    conn.close()

    return results

def find_similar_with_threshold(query, threshold):
    # Calc query embedding
    eingabe_embedding = encode_embedding(query)

    # FAISS-Index
    index = faiss.read_index("gerichte.index")
    ids = np.load("gerichte_ids.npy")

    # Search similar
    D, ind = index.search(eingabe_embedding, len(ids))
    ind = ind[0]
    D = D[0]

    # Meal name
    conn = sqlite3.connect("gerichte.db")
    c = conn.cursor()
    results = []
    print(threshold)
    for i, distance in zip(ind, D):
        similarity = 1 - distance
        if similarity >= threshold:
            c.execute("SELECT name FROM gerichte WHERE id = ?", (int(ids[i]),))
            results.append((c.fetchone()[0], float(similarity)))
    conn.close()
    return results

if __name__ == "__main__":
    print(sys.argv)
    if(len(sys.argv) < 2):
        exit(1)
    else:
        print(find_similar_with_threshold(sys.argv[1], float(sys.argv[2]) / 100.0 ))
