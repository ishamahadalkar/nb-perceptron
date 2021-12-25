##############
# Name: Isha Mahadalkar
# Email: imahadal@purdue.edu
# PUID: 0030874031
# Date: 11/10/2020

# python imahadal-average.py titanic-train.data titanic-train.label titanic-test.data titanic-test.label
# python imahadal-average.py sample.data sample.label sample.data sample.label
# 

import numpy as np
import sys
import os
import pandas as pd
import math
import pprint
import random

# Helper Functions 
# Function to read in the attribute data from the csv file where the missing valus are populated with the mode     
def read_data(filename):
    file = open(filename)
    data_frame = pd.read_csv(file, delimiter= ',', na_values = 'NaN', index_col=None, engine='python')
    data_frame.columns = ["PClass", "Sex", "Age", "Fare", "Embarked", "Relatives", "isAlone"]
    
    attr_notnull = data_frame.dropna()
    mode_df = attr_notnull.mode()

    data_frame['PClass'].fillna(mode_df['PClass'][0], inplace = True)
    data_frame['Sex'].fillna(mode_df['Sex'][0], inplace = True)
    data_frame['Age'].fillna(mode_df['Age'][0], inplace = True)
    data_frame['Fare'].fillna(mode_df['Fare'][0], inplace = True)
    data_frame['Embarked'].fillna(mode_df['Embarked'][0], inplace = True)
    data_frame['Relatives'].fillna(mode_df['Relatives'][0], inplace = True)
    data_frame['isAlone'].fillna(mode_df['isAlone'][0], inplace = True)
    
    return data_frame

# Function to read in the label attribute data from the csv file       
def read_label_data(filename):
    file = open(filename)
    data_frame = pd.read_csv(file, delimiter= ',', index_col=None, engine='python')
    data_frame.columns = ["Survived"]
    
    # Changing the Survived labels 0 to -1
    # data_frame = data_frame.replace([0], -1)
   
    return data_frame 

# Function to split the training data frame 
def split_data_frame(train_data):
    # Splitting the data frame     
    label_data = train_data[['Survived']].copy()
    label_data.columns = ['Survived']
    attr_data = train_data.copy()
    del attr_data['Survived']
    
    return attr_data, label_data

# Function to combine the attribute data and label data
def combine_data_frame(attr_data, label_data):
    # Concatentating the data frames
    train_data = pd.concat([attr_data, label_data], axis=1)
    return train_data

# Function to calculate the accuracy of the prediction 
def calc_accuracy(test, pred):
    res = 0
    for i in range(len(test)):
        if test[i] == pred[i]:
            res = res + 1  
    return (res / float(len(test)))

# Function to divide the training data set into two random data points at a random percent
# Returns train and test split data  
def divide_data(attr_data, label_data, percent):    
    
    # Creating a data frame with the two files combined 
    train_data = combine_data_frame(attr_data, label_data)   
    shuffled_data = train_data.sample(frac=1)
    
    train_split, test_split = np.split(shuffled_data, [int(percent*len(shuffled_data))])
    
    return train_split, test_split

# Make a prediction for each row by doing the dot product
def make_prediction(row, weights, bias):
    # Calculating the sum of the dot products i.e wi * xi 
    temp = 0.0
    index = 0
    for att in row:
        if att == "Survived" or index > (len(weights) - 1):
            break
        temp += weights[index] * row.iloc[0][att]
        index += 1

    # Adding the bias 
    temp += bias
    
    if temp >= 0.0:
        prediction = 1
    else:
        prediction = 0
    
    return prediction

def make_baseline_prediction(train_label_data, test_label_data):
    print("FOR BASELINE PREDICTIONS")
    # Calculating the mode for the label data 
    mode_df = train_label_data.mode()
    freq = mode_df["Survived"][0]
    print("The most frequent class label is: %d" % freq)
    
    predictions = []
    for i in range(len(test_label_data)):
        predictions.append(freq)
    
    # Calculating the test accuracy
    accuracy = calc_accuracy(np.asarray(test_label_data), predictions)
    hinge_loss = max(0, 1 - accuracy)
    print("Baseline error=%0.4f" % (hinge_loss))
    print("Test Accuracy=%0.4f" % (accuracy))
    

def apply_cross_validation(train_attr_data, train_label_data):
    list_weights = []
    list_bias = []
    list_hinge_loss = []
    list_accuracy = []
    
    div_list = [0.01, 0.1, 0.5]
    
    for div in div_list:
        print("For percent: %f" % div)
        for i in range(10):
            # Dividing the training data for k-fold validation
            temp_train, temp_test = divide_data(train_attr_data, train_label_data, div)
            
            # Finding the train and testing attr and label data
            test_attr, test_label = split_data_frame(temp_test)
            train_attr, train_label = split_data_frame(temp_train)
            
            # Train the model
            weights, bias, hinge_loss = train_perceptron(train_attr, train_label)
            list_weights.append(weights)
            list_bias.append(bias)
            list_hinge_loss.append(hinge_loss)
            
            # Test 
            predictions = test_perceptron(test_attr, test_label, weights, bias)
            accuracy = calc_accuracy(np.asarray(test_label), predictions)
            list_accuracy.append(accuracy)
        
        print_analysis(list_hinge_loss, list_accuracy)
    return list_weights, list_bias, list_accuracy

# Method to print the k fold analysis 
def print_analysis(list_hinge_loss, list_accuracy):
    # Printing the analysis 
    length = len(list_accuracy)

    print("MEAN HINGE LOSS=%0.4f" % (sum(list_hinge_loss) / float(length)))
    print("MEAN Test Accuracy=%0.4f" % (sum(list_accuracy) / float(length)))
    print("")
    

# Main function to learn the Perceptron model 
# Which returns the weights vector and the bias
def train_perceptron(train_attr_data, train_label_data):   
    # Combining the two data frames to get a training data frame
    train_data = combine_data_frame(train_attr_data, train_label_data)
    num_rows = len(train_attr_data)
    num_cols = train_attr_data.shape[1]
    
    # Shuffling the data
    train_data = train_data.sample(frac=1)
    
    
    # Initializing the weights vector and the bias
    weights = [0.0] * (num_cols)
    avg_weights = [0.0] * (num_cols)
    
    bias = 0.0
    avg_bias = 0.0
    
    max_iterations = 6
    learning_rate = 0.05
    
    # Make a prediction for each row in the training data 
    count = 0
    iter_n = 0
    error = 0
    list_hinge_loss = []
    while (iter_n < max_iterations): 
        iter_n += 1
        sum_error = 0.0
        for index, row in train_data.iterrows():
            count += 1 
            row_t = pd.DataFrame(row).T
            # Setting the columns 
            row_t.columns = ["PClass", "Sex", "Age", "Fare", "Embarked", "Relatives", "isAlone", "Survived"]
    
            prediction = make_prediction(row_t, weights, bias)
            error = row_t.iloc[0]["Survived"] - prediction 
            sum_error += error**2
                
            # Updating the weights vector and the bias only if the prediction is wrong
            # Also updating the avg weigts and bias
            i = 0
            for att in row_t:
                if (att == "Survived"):
                    break
                weights[i] += learning_rate * error * row_t.iloc[0][att]
                avg_weights[i] += learning_rate * error * row_t.iloc[0][att] * count
                i += 1
                                    
            bias += learning_rate * error
            avg_bias += learning_rate * error * count
            
            t_loss = 0
            loss = 0
            # Calculating the hinge loss
            if error != 0:        
                i = 0        
                for att in row_t:
                    if (att == "Survived"):
                        break
                    t_loss += weights[i] * row_t.iloc[0][att]
                    i += 1
                t_loss *= -row_t.iloc[0]["Survived"]
                loss = max(0, t_loss)            
            list_hinge_loss.append(loss)
    
    for i in range(len(weights)):
            weights[i] -= avg_weights[i] / count

    bias -= avg_bias / count
    # Hinge Loss    
    hinge_loss = sum(list_hinge_loss) / float(len(list_hinge_loss))
    return weights, bias, hinge_loss  

    
# Function to test the Perceptron model 
# Which returns the predictions array
def test_perceptron(test_attr_data, test_label_data, weights, bias):
    
    # Combining the two data frames to get a training data frame
    test_data = combine_data_frame(test_attr_data, test_label_data)

    num_rows = len(test_data)
    num_cols = test_data.shape[1]   
    
    predictions = []
    
    for index, row in test_data.iterrows():
        row_t = pd.DataFrame(row).T
        # Setting the columns 
        row_t.columns = ["PClass", "Sex", "Age", "Fare", "Embarked", "Relatives", "isAlone", "Survived"]
            
        # Make a prediction for each row and add it to the array 
        test_predict = make_prediction(row_t, weights, bias)
        predictions.append(test_predict)
    
    return predictions

if __name__ == "__main__":
    # Parsing the arguments stored in sys.argv
    
    training_attr = sys.argv[1]
    training_label = sys.argv[2]
    testing_attr = sys.argv[3]
    testing_label = sys.argv[4]
    
    # Reading the training attribute and test data 
    train_attr_data = read_data(training_attr)
    train_label_data = read_label_data(training_label)
    
    # Reading the testing attribute and test data 
    test_attr_data = read_data(testing_attr)
    test_label_data = read_label_data(testing_label)
    
    # UNCOMMENT to do cross validation
    # list_weights, list_bias, list_accuracy = apply_cross_validation(train_attr_data, train_label_data)
    
    # UNCOMMENT both line to do baseline analysis 
    # make_baseline_prediction(train_label_data, test_label_data)
    # sys.exit(0)
        
    # Train and test the Perceptron Model
    weights, bias, hinge_loss = train_perceptron(train_attr_data, train_label_data)
    predictions = test_perceptron(test_attr_data, test_label_data, weights, bias)
    
    # Calculating the test accuracy
    accuracy = calc_accuracy(np.asarray(test_label_data), predictions)
    print("HINGE LOSS=%0.4f" % (hinge_loss))
    print("Test Accuracy=%0.4f" % (accuracy))
    