import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Load and prepare data
data = pd.read_csv('SmartCrop-Dataset.csv')
X = data.drop('label', axis=1)
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Save model
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model trained and saved successfully!')