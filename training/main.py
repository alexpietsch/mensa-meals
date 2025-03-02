from training.train_model import train_model
from training.meals_similarity import calc_meal_similarity
from training.prepare_training_data import prepare_data


def main():
    # calc_meal_similarity()
    prepare_data()
    train_model()

if __name__ == "__main__":
    main()
