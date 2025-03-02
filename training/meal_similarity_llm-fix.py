import os
from pathlib import Path
import sys
import pandas as pd
from itertools import combinations
from tqdm.asyncio import tqdm
import aiohttp
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import re
import asyncio

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
MAX_CONCURRENCY = 8  # Number of parallel batches

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

def parse_llm_response(response_text, original_batch):
    """Parse the LLM's CSV response and match it to the original batch
    
    Args:
        response_text: String response from the LLM
        original_batch: List of tuples (meal1, meal2, score) that were sent to the LLM
        
    Returns:
        List of tuples (meal1, meal2, adjusted_score) matched to the original batch
    """
    import csv
    from io import StringIO
    
    results = []
    # Organize original batch into a lookup dictionary
    original_dict = {}
    for meal1, meal2, score in original_batch:
        # Store both directions for easier lookup
        original_dict[(meal1, meal2)] = score
        original_dict[(meal2, meal1)] = score
    
    # First attempt: Use proper CSV parser
    try:
        csv_reader = csv.reader(StringIO(response_text))
        for row in csv_reader:
            if not row or len(row) < 3 or row[0].startswith("Meal 1"):
                continue  # Skip empty rows or header
                
            meal1, meal2, similarity_str = row[0], row[1], row[2]
            try:
                similarity = float(similarity_str)
                # Check if this pair exists in our original batch
                if (meal1, meal2) in original_dict:
                    results.append((meal1, meal2, similarity))
                elif (meal2, meal1) in original_dict:
                    results.append((meal2, meal1, similarity))
            except ValueError:
                # Not a valid float, continue to next row
                pass
    except Exception as e:
        print(f"CSV parsing failed: {str(e)}")
    
    # Second attempt: Process line by line if CSV parsing didn't work well
    if len(results) < len(original_batch):
        # Create lookup sets for more efficient matching
        result_pairs = {(r[0], r[1]) for r in results}
        result_pairs.update({(r[1], r[0]) for r in results})
        
        for line in response_text.strip().split('\n'):
            # Skip empty lines or headers
            if not line.strip() or line.startswith("Meal 1,Meal 2,Similarity"):
                continue
            
            # Try to match this line to an original pair
            for (orig_meal1, orig_meal2), orig_score in original_dict.items():
                # Skip if we already matched this pair
                if (orig_meal1, orig_meal2) in result_pairs or (orig_meal2, orig_meal1) in result_pairs:
                    continue
                
                # Check if both meals are in the line
                if orig_meal1 in line and orig_meal2 in line:
                    # Extract all numbers from the line that could be similarity scores (0.0-1.0)
                    matches = re.findall(r'0?\.\d+', line)
                    if matches:
                        # Take the last number as the similarity score
                        try:
                            similarity = float(matches[-1])
                            if 0 <= similarity <= 1.0:  # Validate it's a proper similarity score
                                results.append((orig_meal1, orig_meal2, similarity))
                                result_pairs.add((orig_meal1, orig_meal2))
                                result_pairs.add((orig_meal2, orig_meal1))
                                break
                        except ValueError:
                            pass
    
    # Ensure we haven't missed any pairs from the original batch
    processed_pairs = {(m1, m2) for m1, m2, _ in results}
    processed_pairs.update({(m2, m1) for m1, m2, _ in results})
    
    for meal1, meal2, score in original_batch:
        if (meal1, meal2) not in processed_pairs and (meal2, meal1) not in processed_pairs:
            # If we missed a pair, add it with its original score
            results.append((meal1, meal2, score))
    
    # Print summary stats
    print(f"Successfully parsed {len(results)} meal pairs out of {len(original_batch)}")
    
    return results

async def enhance_with_llm(meal_pairs, batch_size=LLM_BATCH_SIZE):
    """Enhance similarity scores with LLM analysis
    
    Args:
        meal_pairs: List of tuples (meal1, meal2, score)
        batch_size: Number of meal pairs to send to LLM in one batch
        
    Returns:
        List of tuples (meal1, meal2, adjusted_score)
    """
    print("MEAL PAIRS", len(meal_pairs))
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
Meal 1,Meal 2,Similarity

Where:
- Meal descriptions are exactly as provided in the input
- Similarity is a float value between 0.0 and 1.0"""
    
    enhanced_results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    async def process_batch(batch_indices_and_pairs):
        async with semaphore:
            batch_prompts = []
            batch_indices = []
            
            # Prepare batch
            for idx, pair in batch_indices_and_pairs:
                meal1, meal2, score = pair
                prompt = f"'{meal1}' and '{meal2}' (current score: {score:.4f})\n"
                batch_prompts.append(prompt)
                batch_indices.append(idx)
            
            # Send entire batch to GPU at once
            async with aiohttp.ClientSession() as session:
                tasks = [
                    session.post(
                        OLLAMA_API,
                        json={
                            "model": OLLAMA_MODEL,
                            "system": system_prompt,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.1}
                        }
                    ) for prompt in batch_prompts
                ]
                batch_responses = await asyncio.gather(*tasks)
            
            # Process responses
            for i, (response, idx) in enumerate(zip(batch_responses, batch_indices)):
                if response.status == 200:
                    result_text = await response.json()
                    enhanced_results.extend(parse_llm_response(result_text["response"].strip(), [meal_pairs[idx]]))
                else:
                    print(f"Error: {response.status} - {await response.text()}")
                    enhanced_results.append(meal_pairs[idx])
    
    # Create batches with indices to maintain order
    batches = []
    for i in range(0, len(meal_pairs), batch_size):
        batch = [(idx, pair) for idx, pair in enumerate(meal_pairs[i:i + batch_size])]
        batches.append(batch)
    
    # Process all batches concurrently with progress bar
    await tqdm.gather(*[process_batch(batch) for batch in batches], 
                     desc="Enhancing with LLM", total=len(batches))
    
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
        
        # First, sort results by similarity
        sorted_results = sorted(results, key=lambda x: x[2])
        
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
            range_results = [r for r in sorted_results if lower <= r[2] < upper]
            # Take a sample from each range, proportional to its size but with caps
            sample_size = min(max(int(len(range_results) * 0.1), 10), 500)  # At least 10, max 500
            llm_input.extend(range_results[:sample_size])
        
        # Enhance results with LLM
        enhanced_results = asyncio.run(enhance_with_llm(llm_input))
        
        # Create mapping from meal pairs to new similarity scores
        similarity_map = {}
        for m1, m2, score in enhanced_results:
            # Store both directions for easier lookup
            similarity_map[(m1, m2)] = score
            similarity_map[(m2, m1)] = score
        
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