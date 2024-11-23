import pandas as pd

# Load dataset
data = pd.read_csv(r"C:\\Users\\badhe\\Downloads\\emails.csv")

# Ensure required columns exist
required_columns = ["Message_body"]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# Adjust columns
data = data.rename(columns={"Message_body": "email_text"})

# Handle missing data
data = data.dropna(subset=["email_text"])

# Add a default label if not present
if "label" not in data.columns:
    data["label"] = 0

# Ensure only necessary columns exist
data = data[["email_text", "label"]]

# Log completion
print("Dataset loaded and processed successfully.")
print(data.head())

