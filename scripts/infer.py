import pickle
import pandas as pd

MODEL_PATH = "models/product_classifier.pkl"

def predict(title):
    # Load the model and vectorizer
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)  # Correct usage: pass the file object to pickle.load()
        clf = data["model"]
        vectorizer = data["vectorizer"]

    # Transform input
    X = vectorizer.transform([title])
    prediction = clf.predict(X)[0] 

    # Load category mapping
    with open("data/processed/category_mapping.json", "r") as f:
        import json
        category_mapping = json.load(f)
    reverse_mapping = {v: k for k, v in category_mapping.items()}

    # Output category
    return reverse_mapping[prediction]

if __name__ == "__main__":
    sample_title = "freestanding eeeeeeeFreezer1233333333333"
    print(f"Predicted category: {predict(sample_title)}")
