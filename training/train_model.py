import torch
import pickle
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer, 
    losses
)

output_model_name = "mensa_meal_model_finetuned"
base_model_name = "sentence-transformers/all-MiniLM-L6-v2"

def train_model():
    with open("train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    
    model = SentenceTransformer(base_model_name, device='cuda')

    train_loss = losses.CosineSimilarityLoss(model)
    
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        show_progress_bar=True,
        output_path=output_model_name
    )

    print(f"Model training completed and saved to '{output_model_name}'.")