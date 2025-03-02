import os
from pathlib import Path
import sys
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from merge import merge_files
from transform_input import normalize_and_transform

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
BATCH_SIZE = 128

root_dir = Path(__file__).resolve().parent.parent

def is_potential_match(meal1, meal2, threshold=0.5):
    ratio = SequenceMatcher(None, meal1, meal2).ratio()
    return ratio >= threshold

def get_similarity_batch(meal_pairs):
    meal_texts = [pair[0] for pair in meal_pairs] + [pair[1] for pair in meal_pairs]
    embeddings = model.encode(meal_texts, convert_to_tensor=True)
    similarities = util.cos_sim(embeddings[:len(meal_pairs)], embeddings[len(meal_pairs):]).diagonal()
    return similarities.cpu().tolist()

def calc_meal_similarity():
    input_csv = os.path.join(root_dir,"gerichte_anzahl-py-normalized.csv")
    merge_files()
    normalize_and_transform()
    df = pd.read_csv(input_csv, sep=';')
    meals = df['Gericht'].tolist()
    
    meal_combinations = [(m1, m2) for m1, m2 in combinations(meals, 2) if is_potential_match(m1, m2, 0.5)]
    total_lines = len(meal_combinations)
    results = []
    
    with tqdm(total=total_lines, desc="Processing meal similarities") as pbar:
        for i in range(0, total_lines, BATCH_SIZE):
            batch = meal_combinations[i:i + BATCH_SIZE]
            similarities = get_similarity_batch(batch)
            results.extend([(batch[j][0], batch[j][1], similarities[j]) for j in range(len(batch))])
            pbar.update(len(batch))
    
    result_df = pd.DataFrame(results, columns=['Meal 1', 'Meal 2', 'Similarity'])
    result_df.to_csv('meal_similarities.csv', index=False)
    print("Similarity calculation completed. Results saved to 'meal_similarities.csv'.")

if __name__ == "__main__":
    calc_meal_similarity()
