import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from core import BASE_DIR

def train_title_classifier():
    # Load the data
    data_path = os.path.join(BASE_DIR, "data", "processed", "training_set_v1.jsonl")
    print("Loading data...")
    df = pd.read_json(data_path, lines=True)

    # Prepare Features (X) and Labels (y)
    # We convert the boolean 'is_bold' into 1/0 for the math engine
    X = df[['line_index', 'y_position', 'font_size', 'is_bold']].copy()
    X['is_bold'] = X['is_bold'].astype(int)
    
    y = df['label'].apply(lambda x: 1 if x == "TITLE" else 0)

    # Split into Training (80%) and Testing (20%) sets
    # This ensures we test the model on data it hasn't seen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and Train the Random Forest
    print(f"Training on {len(X_train)} lines...")
    # class_weight='balanced' helps the model not ignore our rare Title lines
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the results
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    # Check Feature Importance
    print("\n--- Feature Importance ---")
    for name, importance in zip(X.columns, model.feature_importances_):
        print(f"{name}: {importance:.4f}")

    # Save the model for later use
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "title_classifier_rf.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_title_classifier()