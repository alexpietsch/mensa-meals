import os
from pathlib import Path
import sys
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import requests
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import re

# Setup paths
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Import your existing preprocessing functions
from merge import merge_files
from transform_input import normalize_and_transform

# Parameters
BATCH_SIZE = 128  # Batch size for GPU processing
# LLM parameters
OLLAMA_MODEL = "qwen2.5:14b"  # You can change this to any model you have in Ollama
OLLAMA_API = "http://localhost:11434/api/generate"
# Set to True to use LLM for correction
USE_LLM_ENHANCEMENT = True
# Max number of examples to send to LLM at once (to avoid context limit issues)
LLM_BATCH_SIZE = 30

def is_potential_match(meal1, meal2, threshold=0.5):
    """Pre-filter meal pairs using SequenceMatcher"""
    ratio = SequenceMatcher(None, meal1, meal2).ratio()
    return ratio >= threshold

def get_similarity_batch(meal_pairs):
    """Get similarity scores using GPU"""
    meal_texts = [pair[0] for pair in meal_pairs] + [pair[1] for pair in meal_pairs]
    embeddings = model.encode(meal_texts, convert_to_tensor=True)
    similarities = util.cos_sim(embeddings[:len(meal_pairs)], embeddings[len(meal_pairs):]).diagonal()
    return similarities.cpu().tolist()

def enhance_with_llm(meal_pairs, batch_size=LLM_BATCH_SIZE):
    """Enhance similarity scores with LLM analysis
    
    Args:
        meal_pairs: List of tuples (meal1, meal2, score)
        batch_size: Number of meal pairs to send to LLM in one batch
        
    Returns:
        List of tuples (meal1, meal2, adjusted_score)
    """
    if not USE_LLM_ENHANCEMENT:
        return meal_pairs
    
    # Load the system prompt
    system_prompt = """# System Prompt: German Canteen Meal Similarity Analysis

## Context & Purpose
You are analyzing German canteen meal descriptions to identify when two differently written meal names refer to the same dish. The existing similarity scores from embedding models miss important linguistic patterns specific to German food descriptions.

## Task
Analyze pairs of meal descriptions and determine their true similarity on a scale from 0.0 to 1.0, where:
- 1.0 = Identical meals (despite minor variations in writing)
- 0.0 = Completely different meals

## Important Patterns to Recognize

### Common German Meal Description Issues:
1. **Missing spaces between words**: "auberginengyrosmit" → "auberginen gyros mit"
2. **Word order variations**: "tzatziki tomatenreis" vs "tomatenreis tzaziki"
3. **Stopwords presence/absence**: "mit spätzle" vs "spätzle"
4. **Spelling variations**: "tzatziki" vs "tzaziki"
5. **Singular/plural forms**: "preiselbeerdip" vs "preiselbeeren"
6. **Additional/missing side dishes**: "kartoffelsuppe" vs "kartoffelsuppe brötchen"
7. **Compound meal components**: "backerbsensuppein" should be "backerbsensuppe in"
8. **Partial descriptions**: A shorter description that is fully contained in a longer one

### Special Considerations:
- A meal with an extra side dish (like "brötchen") is still essentially the same meal (≈0.90 similarity)
- Minor spelling variations should have minimal impact on similarity (≈0.95 similarity)
- Word order for ingredients/sides should have minimal impact (≈0.95 similarity)
- Missing spaces that join words incorrectly are typos and should be treated as the same word

## Output Format
Respond ONLY with a CSV format, one row per meal pair:
```
Meal 1,Meal 2,Similarity
```

Where:
- Meal descriptions are exactly as provided in the input
- Similarity is a float value between 0.0 and 1.0"""
    
    enhanced_results = []
    total_batches = (len(meal_pairs) + batch_size - 1) // batch_size
    
    with tqdm(total=len(meal_pairs), desc="Enhancing with LLM") as pbar:
        for i in range(0, len(meal_pairs), batch_size):
            batch = meal_pairs[i:i + batch_size]
            
            # Create prompt for this batch
            user_prompt = "Analyze these meal pairs and their current similarity scores:\n\n"
            for meal1, meal2, score in batch:
                user_prompt += f"'{meal1}' and '{meal2}' (current score: {score:.4f})\n"
            
            try:
                response = requests.post(
                    OLLAMA_API,
                    json={
                        "model": OLLAMA_MODEL,
                        "system": system_prompt,
                        "prompt": user_prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}
                    }
                )
                
                if response.status_code == 200:
                    # Process the response
                    result_text = response.json()["response"].strip()
                    
                    # Parse the CSV lines
                    for line in result_text.split('\n'):
                        # Skip header line if present
                        if line.startswith("Meal 1,Meal 2,Similarity") or not line.strip():
                            continue
                        
                        # Handle commas within the meal descriptions
                        parts = line.split(',')
                        if len(parts) > 3:  # If there are extra commas in the meal names
                            # Assume the last item is the similarity score
                            similarity = float(parts[-1])
                            
                            # Find the position of the last occurrence of the first meal
                            meal1 = batch[0][0]
                            meal2 = batch[0][1]
                            
                            # Try to extract meal1 and meal2 from the line
                            line_text = ','.join(parts[:-1])  # Everything except the last part
                            
                            # Find the meals in the line
                            matches = []
                            for meal1, meal2, _ in batch:
                                if meal1 in line_text and meal2 in line_text:
                                    matches.append((meal1, meal2))
                            
                            if matches:
                                meal1, meal2 = matches[0]
                            else:
                                # If parsing fails, keep the original score
                                print(f"Failed to parse LLM response line: {line}")
                                continue
                        else:
                            meal1, meal2, similarity = parts[0], parts[1], float(parts[2])
                        
                        # Find the original input meal pair
                        for orig_meal1, orig_meal2, orig_score in batch:
                            if (orig_meal1 == meal1 and orig_meal2 == meal2) or \
                               (orig_meal1 == meal2 and orig_meal2 == meal1):
                                enhanced_results.append((orig_meal1, orig_meal2, similarity))
                                break
                        else:
                            # If no match found, use the parsed values
                            enhanced_results.append((meal1, meal2, similarity))
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    # Keep original scores for this batch
                    enhanced_results.extend(batch)
                    
            except Exception as e:
                print(f"LLM enhancement failed: {str(e)}")
                # Keep original scores for this batch
                enhanced_results.extend(batch)
                
            pbar.update(len(batch))
    
    return enhanced_results

def calc_meal_similarity():
    """Calculate meal similarity using GPU and enhance with LLM"""
    # Load the model
    global model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    
    # Prepare input data
    input_csv = os.path.join(root_dir, "gerichte_anzahl-py-normalized.csv")
    df = pd.read_csv(input_csv, sep=';')
    meals = df['Gericht'].tolist()
    
    # Generate combinations
    print("Generating meal combinations...")
    meal_combinations = [(m1, m2) for m1, m2 in combinations(meals, 2) if is_potential_match(m1, m2, 0.5)]
    total_lines = len(meal_combinations)
    print(f"Found {total_lines} potential meal combinations for analysis")
    
    results = []
    
    # Process batches on GPU
    with tqdm(total=total_lines, desc="Processing meal similarities with GPU") as pbar:
        for i in range(0, total_lines, BATCH_SIZE):
            batch = meal_combinations[i:i + BATCH_SIZE]
            similarities = get_similarity_batch(batch)
            results.extend([(batch[j][0], batch[j][1], similarities[j]) for j in range(len(batch))])
            pbar.update(len(batch))
    
    # Save initial results
    initial_df = pd.DataFrame(results, columns=['Meal 1', 'Meal 2', 'Similarity'])
    initial_df.to_csv('meal_similarities_initial.csv', index=False)
    print("Initial similarity calculation completed.")
    
    # Enhance with LLM if enabled
    if USE_LLM_ENHANCEMENT:
        print("Enhancing results with LLM analysis...")
        
        # Sort results by similarity to ensure we get a good mix of examples
        results.sort(key=lambda x: x[2])
        
        # Select a diverse sample based on similarity ranges
        llm_input = []
        
        # Select meals from different similarity ranges
        similarity_ranges = [
            (0.95, 1.0),   # Almost identical
            (0.8, 0.95),   # Very similar
            (0.65, 0.8),   # Somewhat similar
            (0.5, 0.65),   # Borderline
            (0.0, 0.5)     # Not similar
        ]
        
        for lower, upper in similarity_ranges:
            range_results = [r for r in results if lower <= r[2] < upper]
            # Take a sample from each range, proportional to its size
            sample_size = min(int(len(range_results) * 0.1) + 1, 500)  # At least 1, max 500
            llm_input.extend(range_results[:sample_size])
        
        # Enhance results with LLM
        enhanced_results = enhance_with_llm(llm_input)
        
        # Create mapping from meal pairs to new similarity scores
        similarity_map = {(m1, m2): score for m1, m2, score in enhanced_results}
        similarity_map.update({(m2, m1): score for m1, m2, score in enhanced_results})  # Add reverse pairs
        
        # Update all results with new scores where available
        final_results = []
        for meal1, meal2, score in results:
            if (meal1, meal2) in similarity_map:
                final_results.append((meal1, meal2, similarity_map[(meal1, meal2)]))
            else:
                final_results.append((meal1, meal2, score))
        
        # Save final results
        final_df = pd.DataFrame(final_results, columns=['Meal 1', 'Meal 2', 'Similarity'])
        final_df.to_csv('meal_similarities.csv', index=False)
        
        # Also save the subset that was processed by the LLM
        enhanced_df = pd.DataFrame(enhanced_results, columns=['Meal 1', 'Meal 2', 'Similarity'])
        enhanced_df.to_csv('meal_similarities_llm_enhanced.csv', index=False)
        
        print("LLM enhancement completed. Results saved to 'meal_similarities.csv'")
        print(f"Enhanced {len(enhanced_results)} meal pairs with LLM analysis")
    else:
        # Save the initial results as the final results
        initial_df.to_csv('meal_similarities.csv', index=False)
        print("Similarity calculation completed. Results saved to 'meal_similarities.csv'.")

if __name__ == "__main__":
    # merge_files()
    # normalize_and_transform()
    calc_meal_similarity()