import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

#disable pyplot global use warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
dataset = pd.read_csv("C:\\Users\\madih\\OneDrive\\Desktop\\Zrock_P2\\stockdata.csv")

# Preprocess the data
X = dataset.drop(columns=['Date','Trend'])
y = dataset['Trend']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'LogisticRegression.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)

# Set title using Streamlit
st.title('Stock Price Trend Classification')

# Display dataset
st.subheader('Stock Dataset')
st.write(X)

# Display model coefficients
st.subheader('Model Coefficients')
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
st.write(coefficients)

# Plot coefficients as a bar plot
st.subheader('Model Coefficients Visualization')
plt.figure(figsize=(10, 6))
plt.barh(X.columns, model.coef_[0])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Model Coefficients')
st.pyplot()
plt.close()

# Display model evaluation results
st.subheader('Model Evaluation')
st.write('Accuracy:', accuracy)
