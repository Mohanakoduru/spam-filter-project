import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the trained model
model_path = os.path.join(os.getcwd(), 'optimized_spam_filter.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the text cleaning function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

# Define the spam prediction function
def predict_spam(message):
    tokens = clean_text(message)
    p_spam_train = len(train_data[train_data['label'] == 'spam']) / len(train_data)
    p_ham_train = len(train_data[train_data['label'] == 'ham']) / len(train_data)

    log_prob_spam = math.log(p_spam_train)
    log_prob_ham = math.log(p_ham_train)

    for token in tokens:
        log_prob_spam += math.log(cond_prob_train(token, spam=True))
        log_prob_ham += math.log(cond_prob_train(token, spam=False))

    return 'spam' if log_prob_spam > log_prob_ham else 'ham'

# Example usage
test_message = "Congratulations! You've won a free ticket. Call now."
print("Prediction:", predict_spam(test_message))

# Evaluate the model on the test set
true_labels = test_data['label']
predicted_labels = [predict_spam(message) for message in test_data['message']]

# Display the classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=['spam', 'ham'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['spam', 'ham'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()