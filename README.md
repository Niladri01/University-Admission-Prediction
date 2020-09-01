# University-Admission-Prediction

*Aim:* The aim of this project was to train artificial neural network model to perform regression tasks,perform exploratory data analysis, understand the theory and intuition behind regression models and train them in Scikit Learn and understand the difference between various regression models KPIs such as MSE, RMSE, MAE, R2, adjusted R2.

## Theroretical Overview

*Decision Tress:* Decision Tress are a non-parametric supervised learning method used for classification and regression. They are yes or no model. If the answer is yes, we follow the left hand branch and ask another question. If the answer is no, we follow right hand branch and ask a potentilly different question.

*Random Forest:* Random Forest is a supervised machine learning algorithm consisting many decision trees. The *forest* it builds, is an ensemble of decision trees, usually trained with the “bagging” method.

*Artificial Neural Network:* ANN, usually called neural networks, is the component of artificial intelligence that is meant to simulate the functioning of a human brain. Deep Learning algorithms use neural networks to find associations between a set of inputs and outputs. 

## For Data Modelling  
       import numpy as np
       import pandas as pd

## For Visualization 
       import seaborn as sns 
       import matplotlib.pyplot as plt 

## For Linear Regression
       from sklearn.linear_model import LinearRegression
       from sklearn.metrics import mean_squared_error, accuracy_score

## For Decision Tree
       from sklearn.tree import DecisionTreeRegressor
                    
## For Random Forest
       from sklearn.ensemble import RandomForestRegressor

## For Artificial Neural Network
       import tensorflow as tf
       from tensorflow import keras
       from tensorflow.keras.layers import Dense, Activation, Dropout
       from tensorflow.keras.optimizers import Adam
                  
## Name of the dataset : 
       admission_predict.csv

## Size of the dataset : 
       (500, 8)

## Tasks:
       Loading the data
       Perform Exploratory Data Analysis
       Perform Data Visualization
       Create training and test dataset
       Train and evaluate a linear regression model
       Train and evaluate an Artificial Neural network(ANN)
       Train and evaluate a Decision Tree and random Forest
       Calculate regression model KPI's

## For Data Visualization, I used:
       line plot
       histogram
       pairplot
       heatmap
