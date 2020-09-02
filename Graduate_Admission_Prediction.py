
# coding: utf-8

# # IMPORTING LIBRARIES AND DATASET

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:

df_admission = pd.read_csv("Admission_Predict.csv")


# In[3]:

df_admission.head()


# Let's drop the serial no.

# In[4]:

df_admission.drop("Serial No.", axis = 1, inplace = True)


# # EXPLORATORY DATA ANALYSIS

# checking the null values

# In[5]:

df_admission.isnull()


# In[6]:

df_admission.info()


# In[7]:

df_admission.describe()


# Grouping by University Rating 

# In[8]:

df_university_rating = df_admission.groupby("University Rating").mean()
df_university_rating


# # DATA VISUALIZATION

# In[9]:

df_admission.hist(bins = 30, figsize = (20, 20), color = "r")
plt.show()


# In[10]:

sns.pairplot(df_admission)
plt.show()


# In[11]:

corr_matrix = df_admission.corr()
plt.figure(figsize = (20, 8))
sns.heatmap(corr_matrix, annot = True)
plt.show()


# # TRAINING AND TESTING DATASET

# In[12]:

df_admission.columns


# In[13]:

X = df_admission.drop("Chance of Admit", axis = 1)


# In[14]:

y = df_admission["Chance of Admit"]


# In[15]:

X.shape


# In[16]:

y.shape


# In[17]:

X = np.array(X)
y = np.array(y)


# In[18]:

y = y.reshape(-1, 1)
y.shape


# Scaling the data before training the model

# In[19]:

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)


# In[20]:

scaler_y = StandardScaler()
y = scaler_x.fit_transform(y)


# Spliting the data in to test and train sets

# In[21]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)


# # TRAIN AND EVALUATE A LINEAR REGRESSION MODEL

# In[22]:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


# In[23]:

LinearRegression_model = LinearRegression()
LinearRegression_model.fit(X_train, y_train)


# In[24]:

accuracy_LinearRegression = LinearRegression_model.score(X_test, y_test)
accuracy_LinearRegression


# # TRAIN AND EVALUATE AN ARTIFICIAL NEURAL NETWORK

# In[25]:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam


# In[26]:

ANN_model = keras.Sequential()
ANN_model.add(Dense(50, input_dim = 7))
ANN_model.add(Activation("relu"))
ANN_model.add(Dense(150))
ANN_model.add(Activation("relu"))
ANN_model.add(Dropout(0.5))
ANN_model.add(Dense(150))
ANN_model.add(Activation("relu"))
ANN_model.add(Dropout(0.5))
ANN_model.add(Dense(50))
ANN_model.add(Activation("linear"))
ANN_model.add(Dense(1))
ANN_model.compile(loss = "mse", optimizer = "adam")
ANN_model.summary()


# In[27]:

ANN_model.compile(optimizer = "Adam", loss = "mean_squared_error")


# In[28]:

epochs_hist = ANN_model.fit(X_train, y_train, epochs = 100, batch_size = 20, validation_split = 0.2)


# In[29]:

result = ANN_model.evaluate(X_test, y_test)
accuracy_ANN = 1 - result
print("Accuracy : {}".format(accuracy_ANN))


# In[30]:

epochs_hist.history.keys()


# In[31]:

plt.figure(figsize = (20, 10))
plt.plot(epochs_hist.history["loss"])
plt.title("Model Loss Progress During Training")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend(["Training Loss"])
plt.show()


# # TRAIN AND EVALUATE A DECISION TREE AND RANDOM FOREST MODELS

# Decision tree builds regression or classification models in the form of a tree structure. 
# Decision tree breaks down a dataset into smaller subsets while at the same time an associated decision tree is incrementally developed. 
# The final result is a tree with decision nodes and leaf nodes.

# In[32]:

from sklearn.tree import DecisionTreeRegressor
DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(X_train, y_train)


# In[33]:

accuracy_DecisionTree = DecisionTree_model.score(X_test, y_test)
accuracy_DecisionTree


# Many decision Trees make up a random forest model which is an ensemble model. 
# Predictions made by each decision tree are averaged to get the prediction of random forest model.
# A random forest regressor fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 

# In[34]:

from sklearn.ensemble import RandomForestRegressor
RandomForest_model = RandomForestRegressor(n_estimators = 100, max_depth = 10)
RandomForest_model.fit(X_train, y_train)


# In[35]:

accuracy_RandomForest = RandomForest_model.score(X_test, y_test)
accuracy_RandomForest


# # REGRESSION MODEL KPIs

# In[36]:

plt.figure(figsize = (20, 10))
y_predict = LinearRegression_model.predict(X_test)
plt.plot(y_test, y_predict, "^", color = "r")
plt.show()


# In[37]:

y_predict_original1 = scaler_y.fit_transform(y_predict)
y_test_original1 = scaler_y.fit_transform(y_test)


# In[38]:

y_predict_original = scaler_y.inverse_transform(y_predict_original1)
y_test_original = scaler_y.inverse_transform(y_test_original1)


# In[39]:

plt.figure(figsize=(20, 10))
plt.plot(y_test_original, y_predict_original, "^", color = "r")
plt.show()


# In[40]:

k = X_test.shape[1]
n = len(X_test)
n


# In[41]:

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_original, y_predict_original)),".3f"))
MSE = mean_squared_error(y_test_original, y_predict_original)
MAE = mean_absolute_error(y_test_original, y_predict_original)
r2 = r2_score(y_test_original, y_predict_original)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print("RMSE =", RMSE, "\nMSE =", MSE, "\nMAE =", MAE, "\nR2 =", r2, "\nAdjusted R2 =", adj_r2) 

