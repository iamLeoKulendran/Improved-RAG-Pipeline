import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from langchain_community.vectorstores import Chroma
from get_embedding import get_embedding_function
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Function to save logs
def save_log(log):
    with open('logs.txt', 'a') as f:
        f.write(f"{log}\n")

# Function to extract features
def extract_features(query_text, db, k=5):
    results = db.similarity_search_with_score(query_text, k=k)
    scores = [score for _, score in results]

    # Extract features
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    score_variance = np.var(scores)
    num_docs = len(scores)
    log = [query_text, max_score, score_variance, num_docs]
    print(log)
    save_log(log)  # Save log to file
    return [avg_score, max_score, score_variance, num_docs]

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# Initialize the database
print("Initializing the database...")
CHROMA_DB_PATH = "database"
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

# List to store feature vectors and labels
data = []

# Extract features for each query
print("Extracting features for each query...")
for index, row in df.iterrows():
    query = row['Query']
    label = row['Label']
    features = extract_features(query, db)
    data.append(features + [1 if label == 'Out-of-Scope' else 0])

# Create a DataFrame with the features and labels
print("Creating DataFrame with features and labels...")
feature_df = pd.DataFrame(data, columns=["avg_score", "max_score", "score_variance", "num_docs", "label"])

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X = feature_df[["avg_score", "max_score", "score_variance", "num_docs"]]
y = feature_df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# Print training completion
print("Model training completed.")

# Predict and evaluate
print("Predicting and evaluating...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
print("Saving the model...")
joblib.dump(model, 'oos_model.pkl')
print("Model saved as 'oos_model.pkl'")

# Plotting
print("Generating and saving plots...")

# Plot feature distributions
plt.figure(figsize=(14, 8))
for i, col in enumerate(X.columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=feature_df, x=col, hue='label', kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['In-Scope', 'Out-of-Scope'], yticklabels=['In-Scope', 'Out-of-Scope'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.show()

# Plot ROC curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

print("Plots saved: 'feature_distributions.png', 'confusion_matrix.png', 'roc_curve.png'")