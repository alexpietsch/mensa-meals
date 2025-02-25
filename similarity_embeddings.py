import sqlite3
import sys
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from merge import merge_files
from transform_input import normalize_and_transform


def create_and_save_embeddings():

    model = SentenceTransformer('all-mpnet-base-v2', device="cuda" if torch.cuda.is_available() else "cpu")
    conn = sqlite3.connect("gerichte.db")
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS gerichte (id INTEGER PRIMARY KEY, name TEXT)")
    gerichte_dataframe = pd.read_csv("./gerichte_anzahl-py-normalized.csv", sep=";")

    # Calc Embeddings
    gerichte_dataframe['Embedding'] = list(model.encode(gerichte_dataframe['Gericht'].tolist()).astype('float32'))

    for name in gerichte_dataframe['Gericht']:
        cursor.execute("INSERT INTO gerichte (name) VALUES (?)", (name,))
    conn.commit()

    # IDs from SQLite
    cursor.execute("SELECT id FROM gerichte")
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    # FAISS: Create index and add embeddings
    d = len(gerichte_dataframe['Embedding'][0])  # Dimension der Embeddings
    index = faiss.IndexFlatL2(d)  
    embeddings = np.vstack(gerichte_dataframe['Embedding'].values)
    index.add(embeddings)

    # Save IDs
    np.save("gerichte_ids.npy", np.array(ids))

    # Save FAISS-Index
    faiss.write_index(index, "gerichte.index")


if __name__ == "__main__":
    if(len(sys.argv) == 1):
        create_and_save_embeddings()
    elif(sys.argv[1] == "-c"):
        merge_files()
        normalize_and_transform()
        create_and_save_embeddings()

