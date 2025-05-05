#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.executable)


# In[3]:


print(sys.executable)


# In[1]:


import sys
print(sys.executable)


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


import nltk
print(nltk.data.path)


# In[7]:


#Import Libraries & Load the Dataset
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the SMS Spam Collection dataset (tab-separated file)
data = pd.read_csv(r"C:\Users\mohan\OneDrive\Desktop\ds project\SMSSpamCollection.csv", sep="\t", header=None, names=["label", "message"])

# Quick exploration of the data
print("First 5 rows of the dataset:")
display(data.head())

print("Dataset shape:", data.shape)
print("Label distribution:")
print(data['label'].value_counts())


# In[8]:


#Define the Text Cleaning Function
# Define stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """
    Cleans and tokenizes the input text.
    Steps:
      - Convert text to lowercase
      - Remove non-alphanumeric characters
      - Tokenize the text into words
      - Remove stopwords
      - Apply stemming
    """
    text = text.lowe()  # Lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation & special characters
    tokens = text.split()r  # Tokenize
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

# Test the cleaning function on an example message
sample_message = "Congratulations! You've WON a free ticket. Call now."
print("Original message:", sample_message)
print("Cleaned tokens:", clean_text(sample_message))


# In[9]:


#Preprocess the Data
# Apply the cleaning function to add a new 'tokens' column
data['tokens'] = data['message'].apply(clean_text)

# Display the first 5 messages with their tokens for verification
print("Sample messages with tokens:")
display(data[['message', 'tokens']].head())


# In[10]:


#Calculate Prior Probabilities
# Total number of messages and counts per label
total_messages = len(data)
spam_messages = len(data[data['label'] == 'spam'])
ham_messages = len(data[data['label'] == 'ham'])

# Calculate prior probabilities
p_spam = spam_messages / total_messages
p_ham = ham_messages / total_messages

print(f"P(spam): {p_spam:.4f}")
print(f"P(ham): {p_ham:.4f}")


# In[13]:


#conditional probability
# Initialize counters for spam and ham words
spam_counter = Counter()
ham_counter = Counter()

for idx, row in data.iterrows():
    tokens = row['tokens']
    if row['label'] == 'spam':
        spam_counter.update(tokens)
    else:
        ham_counter.update(tokens)

# Compute total word counts for each category
total_spam_words = sum(spam_counter.values())
total_ham_words = sum(ham_counter.values())

# Create the vocabulary of all unique words
vocabulary = set(list(spam_counter.keys()) + list(ham_counter.keys()))
vocab_size = len(vocabulary)

print("Total words in spam messages:", total_spam_words)
print("Total words in ham messages:", total_ham_words)
print("Vocabulary size:", vocab_size)

# Define a function to compute conditional probability with Laplace smoothing
def cond_prob(word, spam=True):
    if spam:
        return (spam_counter[word] + 1) / (total_spam_words + vocab_size)
    else:
        return (ham_counter[word] + 1) / (total_ham_words + vocab_size)

# Test the conditional probability for a sample word:
sample_word = "free"
print(f"P('{sample_word}' | spam): {cond_prob(sample_word, spam=True):.4f}")
print(f"P('{sample_word}' | ham): {cond_prob(sample_word, spam=False):.4f}")


# In[16]:


#Implement the Naive Bayes Prediction Function
def predict(message):
    """
    Predicts 'spam' or 'ham' for the given message using the computed probabilities.
    """
    tokens = clean_text(message)
    log_prob_spam = math.log(p_spam)
    log_prob_ham = math.log(p_ham)

    for token in tokens:
        log_prob_spam += math.log(cond_prob(token, spam=True))
        log_prob_ham += math.log(cond_prob(token, spam=False))

    return 'spam' if log_prob_spam > log_prob_ham else 'ham'

# Test the prediction function on a sample message
test_message = "congracts! you won 500000."
print("Test message:", test_message)
print("Prediction:", predict(test_message))


# In[17]:


#Recompute Counters on the Training Data
# Split data into training (80%) and testing (20%) sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

print("Training Data shape:", train_data.shape)
print("Testing Data shape:", test_data.shape)


# In[18]:


# Re-compute the counters using only training data
spam_counter_train = Counter()
ham_counter_train = Counter()

for idx, row in train_data.iterrows():
    tokens = row['tokens']
    if row['label'] == 'spam':
        spam_counter_train.update(tokens)
    else:
        ham_counter_train.update(tokens)

total_spam_words_train = sum(spam_counter_train.values())
total_ham_words_train = sum(ham_counter_train.values())
vocabulary_train = set(list(spam_counter_train.keys()) + list(ham_counter_train.keys()))
vocab_size_train = len(vocabulary_train)

print("Training set - Total spam words:", total_spam_words_train)
print("Training set - Total ham words:", total_ham_words_train)
print("Training set - Vocabulary size:", vocab_size_train)

# Define conditional probability function based on training data
def cond_prob_train(word, spam=True):
    if spam:
        return (spam_counter_train[word] + 1) / (total_spam_words_train + vocab_size_train)
    else:
        return (ham_counter_train[word] + 1) / (total_ham_words_train + vocab_size_train)


# In[19]:


#Prediction Function Using Training Data
def predict_train(message):
    """
    Predicts the label for a given message using training data statistics.
    """
    tokens = clean_text(message)
    p_spam_train = len(train_data[train_data['label'] == 'spam']) / len(train_data)
    p_ham_train = len(train_data[train_data['label'] == 'ham']) / len(train_data)

    log_prob_spam = math.log(p_spam_train)
    log_prob_ham = math.log(p_ham_train)

    for token in tokens:
        log_prob_spam += math.log(cond_prob_train(token, spam=True))
        log_prob_ham += math.log(cond_prob_train(token, spam=False))

    return 'spam' if log_prob_spam > log_prob_ham else 'ham'

# Test the training-based prediction function
print("Training-based prediction for test message:", predict_train(test_message))


# In[20]:


#Evaluate the Model on the Test Set
correct_predictions = 0
total_test = len(test_data)
results = []

for idx, row in test_data.iterrows():
    prediction = predict_train(row['message'])
    results.append({
        'message': row['message'],
        'true_label': row['label'],
        'predicted_label': prediction
    })
    if prediction == row['label']:
        correct_predictions += 1

accuracy = correct_predictions / total_test
print(f"Test set accuracy: {accuracy:.4f}")

# Optional: view some prediction results
results_df = pd.DataFrame(results)
display(results_df.head())


# In[21]:


from sklearn.metrics import classification_report

# Create arrays for true and predicted labels from the test set
true_labels = test_data['label']
predicted_labels = [predict_train(m) for m in test_data['message']]

# Display the classification report
print(classification_report(true_labels, predicted_labels))


# In[1]:


import seaborn as sns
print(sns.__version__)


# In[5]:


#Import Libraries & Load the Dataset
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the SMS Spam Collection dataset (tab-separated file)
data = pd.read_csv(r"C:\Users\mohan\OneDrive\Desktop\ds project\SMSSpamCollection.csv", sep="\t", header=None, names=["label", "message"])

# Quick exploration of the data
print("First 5 rows of the dataset:")
display(data.head())

print("Dataset shape:", data.shape)
print("Label distribution:")
print(data['label'].value_counts())


# In[6]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

print("Training Data shape:", train_data.shape)
print("Testing Data shape:", test_data.shape)


# In[8]:


# Import Libraries
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



# In[9]:


# Set up stopwords and the stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """
    Cleans and tokenizes the input text.
    Steps:
      - Convert text to lowercase
      - Remove non-alphanumeric characters
      - Tokenize the text into words
      - Remove stopwords
      - Apply stemming
    """
    text = text.lower()  # Corrected function call
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation & special characters
    tokens = text.split()  # Tokenization
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

# Load your dataset
data = pd.read_csv(r"C:\Users\mohan\OneDrive\Desktop\ds project\SMSSpamCollection.csv", 
                   sep="\t", header=None, names=["label", "message"])

# Verify the data loaded correctly
print("First 5 rows of the dataset:")
display(data.head())
print("Dataset shape:", data.shape)
print("Label distribution:")
print(data['label'].value_counts())


# In[10]:


# Apply text cleaning and get tokens for each message
data['tokens'] = data['message'].apply(clean_text)

# Display sample messages with tokens for verification
print("Sample messages with tokens:")
display(data[['message', 'tokens']].head())


# In[11]:


train_data, test_data = train_test_split(
    data, 
    test_size=0.2, 
    random_state=42, 
    stratify=data['label']
)

print("Training Data shape:", train_data.shape)
print("Testing Data shape:", test_data.shape)


# In[12]:


# Initialize counters for spam and ham in the training data
spam_counter_train = Counter()
ham_counter_train = Counter()

for idx, row in train_data.iterrows():
    tokens = row['tokens']
    if row['label'] == 'spam':
        spam_counter_train.update(tokens)
    else:
        ham_counter_train.update(tokens)

# Compute total word counts and vocabulary size for training data
total_spam_words_train = sum(spam_counter_train.values())
total_ham_words_train = sum(ham_counter_train.values())
vocabulary_train = set(list(spam_counter_train.keys()) + list(ham_counter_train.keys()))
vocab_size_train = len(vocabulary_train)

print("Training set - Total spam words:", total_spam_words_train)
print("Training set - Total ham words:", total_ham_words_train)
print("Training set - Vocabulary size:", vocab_size_train)

# Define the conditional probability function based on training data
def cond_prob_train(word, spam=True):
    if spam:
        return (spam_counter_train[word] + 1) / (total_spam_words_train + vocab_size_train)
    else:
        return (ham_counter_train[word] + 1) / (total_ham_words_train + vocab_size_train)


# In[13]:


def predict_train(message):
    """
    Predicts the label ('spam' or 'ham') for a given message using training data statistics.
    """
    tokens = clean_text(message)
    p_spam_train = len(train_data[train_data['label'] == 'spam']) / len(train_data)
    p_ham_train = len(train_data[train_data['label'] == 'ham']) / len(train_data)

    # Initialize log probabilities with the prior probabilities
    log_prob_spam = math.log(p_spam_train)
    log_prob_ham = math.log(p_ham_train)

    # Sum the log conditional probabilities for each token
    for token in tokens:
        log_prob_spam += math.log(cond_prob_train(token, spam=True))
        log_prob_ham += math.log(cond_prob_train(token, spam=False))

    # Predict the class with higher probability
    return 'spam' if log_prob_spam > log_prob_ham else 'ham'

# Test the predict_train function on a sample message
test_message = "congracts! you won 500000."
print("Training-based prediction for test message:", predict_train(test_message))


# In[14]:


# Get predictions for the test set using predict_train
true_labels = test_data['label']
predicted_labels = [predict_train(message) for message in test_data['message']]

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=['spam', 'ham'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['spam', 'ham'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Print a detailed classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

# Error Analysis: Display misclassified messages
misclassified = test_data.copy()
misclassified['predicted'] = predicted_labels
misclassified = misclassified[misclassified['label'] != misclassified['predicted']]
print("Some misclassified messages:")
display(misclassified[['message', 'label', 'predicted']].head(10))


# In[15]:


# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

# Scikit-learn libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the SMS Spam Collection dataset 
data = pd.read_csv(r"C:\Users\mohan\OneDrive\Desktop\ds project\SMSSpamCollection.csv", 
                   sep="\t", header=None, names=["label", "message"])

# Quick exploration
print("First 5 rows of the dataset:")
display(data.head())
print("Dataset shape:", data.shape)
print("Label distribution:")
print(data['label'].value_counts())


# In[16]:


# Split data into training and test sets
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


# In[17]:


# Create a pipeline with TF-IDF and Multinomial Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),  # we can let the vectorizer clean the text
    ('nb', MultinomialNB())
])

# Define hyperparameters for grid search; 
# ngram_range of (1,2) considers both unigrams and bigrams which often improves performance in text tasks.
parameters = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__max_df': [0.90, 0.95, 1.0],
    'tfidf__min_df': [1, 2],
    'nb__alpha': [0.1, 0.5, 1.0]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    pipeline, 
    param_grid=parameters, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1
)

# Fit grid search on the training data
grid_search.fit(X_train, y_train)

# Display the best parameters and best cross-validated score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy: {:.4f}".format(grid_search.best_score_))


# In[18]:


# Use the best estimator from grid search to make predictions on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate and print test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy: {:.4f}".format(test_accuracy))

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['spam','ham'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['spam', 'ham'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Optimized Model')
plt.show()


# In[19]:


import pickle

# Save the model for future use
with open('optimized_spam_filter.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# To load the model later:
# with open('optimized_spam_filter.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)
#     example_prediction = loaded_model.predict(["Congratulations! You have won a prize."])
#     print("Loaded Model Prediction:", example_prediction)


# In[20]:


import pickle

# Save your best model (from your GridSearchCV)
with open('optimized_spam_filter.pkl', 'wb') as file:
    pickle.dump(best_model, file)


# In[1]:


import flask
print("Flask version:", flask.__version__)


# In[2]:


import importlib.metadata

flask_version = importlib.metadata.version("flask")
print("Flask version:", flask_version)


# In[5]:


import os
import pickle
from flask import Flask

app = Flask(__name__)

# Use os.getcwd() instead of __file__ to determine the current working directory
model_path = os.path.join(os.getcwd(), 'optimized_spam_filter.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)


# In[ ]:


import os
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your saved model. Ensure 'optimized_spam_filter.pkl' is in your current working directory.
model_path = os.path.join(os.getcwd(), 'optimized_spam_filter.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Spam Filter API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload in the form:
      {"message": "Your message text here"}
    And returns a JSON response with the prediction.
    """
    data = request.get_json(force=True)

    # Validate input payload
    if 'message' not in data:
        return jsonify({"error": "Missing 'message' parameter in JSON request"}), 400

    message = data['message']

    # Use your model's pipeline to make a prediction
    prediction = model.predict([message])[0]

    response = {
        "message": message,
        "prediction": prediction
    }
    return jsonify(response)

if __name__ == '__main__':
    # When running in Jupyter Notebook, disable the reloader to avoid SystemExit errors.
    app.run(debug=True, use_reloader=False)


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import os\nimport pickle\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n# Load your saved model. Ensure \'optimized_spam_filter.pkl\' is in your current working directory.\nmodel_path = os.path.join(os.getcwd(), \'optimized_spam_filter.pkl\')\nwith open(model_path, \'rb\') as file:\n    model = pickle.load(file)\n\n@app.route(\'/\')\ndef home():\n    return "Spam Filter API is up and running!"\n\n@app.route(\'/predict\', methods=[\'POST\'])\ndef predict():\n    """\n    Expects a JSON payload in the form:\n      {"message": "Your message text here"}\n    And returns a JSON response with the prediction.\n    """\n    data = request.get_json(force=True)\n\n    # Validate input payload\n    if \'message\' not in data:\n        return jsonify({"error": "Missing \'message\' parameter in JSON request"}), 400\n\n    message = data[\'message\']\n\n    # Use your model\'s pipeline to make a prediction\n    prediction = model.predict([message])[0]\n\n    response = {\n        "message": message,\n        "prediction": prediction\n    }\n    return jsonify(response)\n\nif __name__ == \'__main__\':\n    # Disable the reloader to avoid issues in notebook environments\n    app.run(debug=True, use_reloader=False)\n')


# In[ ]:


git init
git add .
git commit -m "Initial commit of spam filter API"


# In[ ]:




