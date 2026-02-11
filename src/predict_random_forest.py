import os
import joblib
import pandas as pd
from pdf2bibtex.core import RAW_PDF_DIR, BASE_DIR
from pdf_loader import PDFLoader

class TitlePredictor:
    def __init__(self, model_path: str):
        # Load the trained brain
        self.model = joblib.load(model_path)
        print("Model loaded successfully.")

    def predict_title(self, pdf_path: str) -> str:
        # Extract lines from the first page
        loader = PDFLoader(pdf_path)
        lines = loader.get_first_page_lines()
        loader.close()

        if not lines:
            return "No text found in PDF."

        # Convert lines to the format the model expects (X)
        data = []
        for l in lines:
            data.append({
                'line_index': l.line_index,
                'y_position': l.y_position,
                'font_size': l.font_size,
                'is_bold': int(l.is_bold)
            })
        
        X = pd.DataFrame(data)

        # Get probabilities for each line
        # predict_proba returns [prob_of_0, prob_of_1]
        probs = self.model.predict_proba(X)[:, 1]

        # Find the lines with the highest probability
        # We'll take all lines the model is > 50% sure are titles
        title_indices = [i for i, p in enumerate(probs) if p > 0.5]
        
        if not title_indices:
            # Fallback: just take the single highest probability line
            best_idx = probs.argmax()
            title_indices = [best_idx]

        # Join the predicted lines together
        predicted_title = " ".join([lines[i].text for i in title_indices])
        return predicted_title.strip()

if __name__ == "__main__":
    model_file = os.path.join(BASE_DIR, "models", "title_classifier_rf.joblib")
    predictor = TitlePredictor(model_file)

    # Test on a few files from raw folder
    test_files = [f for f in os.listdir(RAW_PDF_DIR) if f.endswith('.pdf')][:5]
    
    for filename in test_files:
        path = os.path.join(RAW_PDF_DIR, filename)
        title = predictor.predict_title(path)
        print(f"\nFILE: {filename}")
        print(f"PREDICTED TITLE: {title}")
        print("-" * 30)