import os
import pandas as pd
from sentence_transformers import InputExample
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

def prepare_data():
    input_csv = os.path.join(root_dir,"meal_similarities.csv")
    df = pd.read_csv(input_csv)

    train_data = [InputExample(texts=[row["Meal 1"], row["Meal 2"]], label=row["Similarity"]) for _, row in df.iterrows()]

    import pickle
    with open("train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)

    print(f"Saved {len(train_data)} training samples.")
