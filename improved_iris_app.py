import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
rf_clf=RandomForestClassifier(n_jobs=-1,n_estimators=100)
rf_clf.fit(X_train,y_train)
log_reg=LogisticRegression(n_jobs=-1)
log_reg.fit(X_train,y_train)
score = svc_model.score(X_train, y_train)
@st.cache()
def prediction(model,SepalLength, SepalWidth, PetalLength, PetalWidth):
  species = model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"
st.sidebar.title('Iris flower species prediction app')
s_length=st.sidebar.slider('sepal length',0.0,10.0)
s_width=st.sidebar.slider('sepal width',0.0,10.0)
p_length=st.sidebar.slider('petal length',0.0,10.0)
p_width=st.sidebar.slider('petal width',0.0,10.0)
classifier=st.sidebar.selectbox('classifier',('svc','lr','rfc'))


if st.sidebar.button('predict'):
	if classifier=='svc':
		species_type=prediction(svc_model,s_length,s_width,p_length,p_width)
		score=svc_model.score(X_train,y_train)
	elif classifier=='lr':
		species_type=prediction(log_reg,s_length,s_width,p_length,p_width)
		score=log_reg.score(X_train,y_train)
	else:
		species_type=prediction(rf_clf,s_length,s_width,p_length,p_width)
		score=rf_clf.score(X_train,y_train)

	st.write('species predicted =',species_type)
	st.write('accuracy = ',score)
	