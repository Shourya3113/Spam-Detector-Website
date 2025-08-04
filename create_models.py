# create_models.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
import os # <-- Import the 'os' module

print("Starting model creation process...")

# --- [FIX] Build a reliable path to the CSV file ---
# This makes the script work regardless of where you run it from.
# It finds the script's own directory and looks for 'spam.csv' inside it.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'spam.csv')
    print(f"Looking for dataset at: {csv_path}")
except NameError:
    # This fallback is for interactive environments like Jupyter notebooks
    csv_path = 'spam.csv'
    print("Running in an interactive environment. Assuming 'spam.csv' is in the current directory.")


# --- 1. Load Data ---
# Load the dataset from the CSV file using the full path we just created.
# We specify encoding='latin-1' because the file has some special characters.
try:
    df = pd.read_csv(csv_path, encoding='latin-1')
    print("Dataset 'spam.csv' loaded successfully.")
except FileNotFoundError:
    print(f"\nERROR: '{csv_path}' not found.")
    print("Please make sure 'spam.csv' is in the same folder as this script.")
    print("You can download the dataset from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset\n")
    exit()


# --- 2. Prepare Data ---
# The dataset has extra columns we don't need. We'll keep only 'v1' (the label) and 'v2' (the message).
df = df[['v1', 'v2']]
# Rename the columns for clarity.
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# Convert labels ('ham'/'spam') to numerical format (0/1).
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print("Data prepared: columns renamed and labels converted to numbers.")


# --- 3. Define Model and Vectorizer ---
# We'll use a simple and effective model for text classification: Multinomial Naive Bayes.
# The TfidfVectorizer will convert our text messages into numerical vectors.
# We create a 'Pipeline' to chain these two steps together. This is best practice.
model_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', strip_accents='unicode')),
    ('classifier', MultinomialNB())
])
print("Model pipeline created (TfidfVectorizer -> MultinomialNB).")


# --- 4. Split Data and Train Model ---
X = df['message']
y = df['label_num']

# Split data into a training set and a testing set (we won't use the test set here, but it's good practice).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model on the data...")
# Train the entire pipeline on the training data.
model_pipeline.fit(X_train, y_train)
print("Model training complete.")


# --- 5. Save the Model and Vectorizer ---
# Now, we save the two essential parts of our pipeline.

# Save the trained model (the classifier part of the pipeline)
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline.named_steps['classifier'], f)
print("Model saved as 'model.pkl'.")

# Save the fitted vectorizer (the vectorizer part of the pipeline)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(model_pipeline.named_steps['vectorizer'], f)
print("Vectorizer saved as 'vectorizer.pkl'.")

print("\nProcess finished successfully!")
