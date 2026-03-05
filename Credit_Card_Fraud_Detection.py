# Generated from: Credit_Card_Fraud_Detection.ipynb
# Converted at: 2026-03-05T11:58:21.326Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# <a href="https://colab.research.google.com/github/random-gau/CODSOFT/blob/main/Credit_Card_Fraud_Detection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>




# Import Libraries


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading Dataset to pandas Dataframe
credit_card_data = pd.read_csv('/content/creditcard.csv')

# Print first 5 rows
credit_card_data.head()

# Print last 5 rows
credit_card_data.tail()

# Dataset Information
credit_card_data.info()

#checking missing values
credit_card_data.isnull().sum()

#distribution of legit transaction ans fraudulant transaction
credit_card_data['Class'].value_counts()



# This is highly unbalanced datset
# 


# 0--> normal transaction
# 1--> fradulant transaction
# 


# seperation the data for analysis
normal = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]




print(normal.shape)
print(fraud.shape)

#statistical measure of the data
normal.Amount.describe()

fraud.Amount.describe()

 # compare values of both transaciton
 credit_card_data.groupby('Class').mean()



# Under sampling
# 
# 
# 
# building a dataset containing similar distibution of normal transaction and fradulent transaction
# 


normal_sample = normal.sample(n=492)



# Concatenating 2 DF
# 


new_dataset = pd.concat([normal_sample, fraud], axis=0)

new_dataset.head()

new_dataset.tail()

new_dataset['Class'].value_counts()



new_dataset.groupby('Class').mean()

# Splitting the data into feature and targets


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


print(X)

print(Y)



# Split the data into training and testing data


X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)



# Model Training
# 
# 
# Logistic Regression


model = LogisticRegression()

#training model with training data
model.fit(X_train , Y_train)



# Model Evaluation
# 


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('ACCURACY ON TRAINING DATA :', training_data_accuracy)



# accuracy on testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('ACCURACY ON TESTING DATA :',testing_data_accuracy)