import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

"""
    In this example, we will use the Pima Indians Diabetes dataset to predict whether or not
    a person has diabetes based on several medical predictor variables.
    The dataset contains 768 instances and 8 attributes. We will preprocess
    the data, split it into training and test sets, train a logistic
    regression model, and evaluate its performance using accuracy 
    and confusion matrix.
"""


# Load the dataset
df = pd.read_csv('diabetes.csv')

# Split the dataset into features and target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Perform data preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion matrix:\n{cm}')
