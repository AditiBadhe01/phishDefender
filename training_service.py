import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os
import json
import falcon

MODEL_PATH = "app/models/phishing_model.pkl"
DATA_PATH = "data/emails.csv"

class TrainModelResource:
    """
    Resource for training the phishing detection model.
    """

    def on_post(self, req, resp):
        """
        Handle POST requests to train the model.
        Accepts `n_estimators` as a parameter.
        """
        try:
            # Parse request body
            raw_data = req.bounded_stream.read()
            params = json.loads(raw_data)

            # Extract model parameters
            n_estimators = params.get("n_estimators", 100)

            # Check if dataset exists
            if not os.path.exists(DATA_PATH):
                resp.status = falcon.HTTP_400
                resp.media = {"error": "Training data not found. Place 'emails.csv' in the data/ folder."}
                return

            # Load dataset
            data = pd.read_csv(DATA_PATH)

            # Ensure required columns exist
            if 'Message_body' not in data.columns:
                resp.status = falcon.HTTP_400
                resp.media = {"error": "'Message_body' column not found in the dataset."}
                return

            # Preprocess data
            data = data.rename(columns={"Message_body": "email_text"})
            data = data.dropna(subset=["email_text"])
            if 'label' not in data.columns:
                data["label"] = 0

            X = data["email_text"]
            y = data["label"]

            # Convert text to numerical features
            vectorizer = CountVectorizer()
            X_vectorized = vectorizer.fit_transform(X)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

            # Train Random Forest model
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Save model and vectorizer
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump((model, vectorizer), MODEL_PATH)

            # Respond with success message and accuracy
            resp.status = falcon.HTTP_200
            resp.media = {"message": "Model trained successfully", "accuracy": accuracy}

        except json.JSONDecodeError:
            resp.status = falcon.HTTP_400
            resp.media = {"error": "Invalid JSON format in request body."}

        except Exception as e:
            resp.status = falcon.HTTP_500
            resp.media = {"error": f"An internal error occurred: {str(e)}"}
