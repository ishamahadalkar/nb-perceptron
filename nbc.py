##############
# Name: Isha Mahadalkar
# Email: imahadal@purdue.edu
# PUID: 0030874031
# Date: 11/10/2020

# python imahadal-nbc.py titanic-train.data titanic-train.label titanic-test.data titanic-test.label
# python imahadal-nbc.py sample.data sample.label sample.data sample.label
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

# Function to calculate the number of instances for Survived from the data frame where x can be 0 or 1
def count_survived(data_frame, x):
    df_temp = data_frame.query('Survived == @x')
    return len(df_temp)

# Function to calculate the number of instances for a given label from the data frame where x can be any unique value 
def count_label(data_frame, label, x):
    query = "{} == @x".format(label)
    df_temp = data_frame.query(query)
    return len(df_temp)

# Function to calculate the number of instances for a given label with the given attr where x can be 0 or 1
# and y can be any instance of the label or the threshold
def count_survived_given_label(data_frame, label, x, y):
    query = "{} == @y".format(label)
    df_temp = data_frame.query(query)
    df_temp2 = df_temp.query('Survived == @x')
    return len(df_temp2)

# Function to calculate the number of instances for a given label with the given attr where x can be 0 or 1
# and y can be any instance of the label or the threshold
# Returns less than and more than values 
def count_survived_given_label_continuous(data_frame, label, x, y):
    query = "{} <= @y".format(label)
    df_temp = data_frame.query(query)
    df_temp2 = df_temp.query('Survived == @x')
    less = len(df_temp2)
    query = "{} > @y".format(label)
    df_temp = data_frame.query(query)
    df_temp2 = df_temp.query('Survived == @x')
    more = len(df_temp2)
    
    return less, more 

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


# Method to print the k fold analysis 
def print_analysis(list_loss, list_squared_loss, analysis):
    # Printing the analysis 
    length = len(list_loss)
    print("MEAN ZERO-ONE LOSS=%0.4f" % (sum(list_loss) / float(length)))
    print("MEAN SQUARED LOSS=%0.4f" % (sum(list_squared_loss) / float(length)))
    
def make_baseline_prediction(train_label_data, test_label_data):
    print("FOR BASELINE PREDICTIONS")
    # Calculating the mode for the label data 
    mode_df = train_label_data.mode()
    freq = mode_df["Survived"][0]
    print("The most frequent class label is: %d" % freq)
    
    predictions = []
    for i in range(len(test_label_data)):
        predictions.append(freq)
   
    # Calculating the test accuracy and zero-one loss
    accuracy = calc_accuracy(np.asarray(test_label_data), predictions)
    loss = 1 - accuracy

    print("")
    print("Basline Error=%0.4f" % (loss))
    print("Test Accuracy=%0.4f" % (accuracy))
    
    
def apply_cross_validation(train_attr_data, train_label_data):
    
    list_prob_survived = [] 
    list_prob_dead = [] 
    list_prob_attr_yes = []
    list_prob_attr_no = []
    
    list_loss = []
    list_squared_loss = []   
    analysis = []
    
    div_list = [0.01, 0.1, 0.5]
    
    for div in div_list:
        print("For percent: %f" % div)
        for i in range(10):
            # Dividing the training data for k-fold validation
            temp_train, temp_test = divide_data(train_attr_data, train_label_data, div)

            # Finding the train and testing attr and label data
            test_attr, test_label = split_data_frame(temp_test)
            train_attr, train_label = split_data_frame(temp_train)

            # Train the Naive Bayes Model
            prob_survived, prob_dead, prob_attr_yes, prob_attr_no = train_NBC(train_attr, train_label)
            list_prob_survived.append(prob_survived)
            list_prob_dead.append(prob_dead)
            list_prob_attr_yes.append(prob_attr_yes)
            list_prob_attr_no.append(prob_attr_no)
        
            predictions, loss, squared_loss = predict_NBC(test_attr, test_label, prob_survived, prob_dead, prob_attr_yes, prob_attr_no)
            list_loss.append(loss)
            list_squared_loss.append(squared_loss)
            accuracy = calc_accuracy(np.asarray(test_label), predictions)
            analysis.append(accuracy)
        
        print_analysis(list_loss, list_squared_loss, analysis)
    

# Main function to learn the NBC model 
# Which returns prob_survived, prob_dead, prob_attr_yes, prob_attr_no
def train_NBC(attr_data, label_data):
    # Creating a training data frame with the two files combined 
    train_data = combine_data_frame(attr_data, label_data)
  
    num_rows = len(train_data)
    
    # Calculating the prior probabilites
    # P(Y = k) or P(C)
    num_survived = count_survived(train_data, 1)
    prob_survived = num_survived / float(num_rows)
    num_dead = count_survived(train_data, 0)
    prob_dead = num_dead / float(num_rows)
    
    # Using a for loop to calculate the conditional probability of every attribute
    # P(Xi = k| Y = k)
    
    # prob_attr[Attr] = [prob_survived for each unique val for that attr]
    prob_attr_yes = {}
    prob_attr_no = {}

    for att in attr_data:
        # Calculating the probability of evidence 
        # P(X1) * P(X2) ... * P(Xn)
               
        # Finding the unique values as a list where k = val_list[i] for each X1..Xn (att)
        val_list = train_data[att].unique().tolist()
        val_list.sort()
        
        # t_yes[unique val] = P(Att = unique val | 1)
        # t_no[unique val] = P(Att = unique val | 0)
        t_yes = {}
        t_no = {}
        
        # Calculate the probabilty of "Survived" or "Dead" for every unique value
        for i in range(len(val_list)):
            # Laplace Smoothing
            # Adding 1 to the numebrator and k (no. of unique values of X) to the denominator
            
            # Using loglikelihood to normalize the data 
            p_s = float((count_survived_given_label(train_data, att, 1, val_list[i]) + 1) / float(num_survived + len(val_list)))
            t_yes[val_list[i]] = math.log(p_s)
            
            p_d = float((count_survived_given_label(train_data, att, 0, val_list[i]) + 1) / float(num_dead + len(val_list)))
            t_no[val_list[i]] = math.log(p_d)
            
                 
        prob_attr_yes[att] = t_yes
        prob_attr_no[att] = t_no
    
    return prob_survived, prob_dead, prob_attr_yes, prob_attr_no

    
def predict_NBC(test_attr, test_label, prob_survived, prob_dead, prob_attr_yes, prob_attr_no):
    # Creating a testing data frame with the two files combined 
    test_data = combine_data_frame(test_attr, test_label)
    
    # Test each row of the test data and predict the label          
    predictions = []
    
    # Calculating Pi = P(Y = True Class Label | X)
    pi = 0.0 
    pi_s = 0.0
    pi_d = 0.0
    sl = 0.0
    loss = 0
    
    # Calculating P(X)
    p_x = 0.0

    for index, row in test_data.iterrows():
        row_t = pd.DataFrame(row).T
        # Setting the columns 
        row_t.columns = ["PClass", "Sex", "Age", "Fare", "Embarked", "Relatives", "isAlone", "Survived"]
           
        prob_test_survived = 0.0
        prob_test_survived += math.log(prob_survived)
        pi_s = prob_test_survived
        prob_test_dead = 0.0
        prob_test_dead += math.log(prob_dead)
        pi_d = prob_test_dead
        
        # Calculating the conditional probability for each attribute 
        # Since, we have a logliklihood we do the sum of the conditional probabilites 
        for att in row_t:
            if att == "Survived":
                break
            # Calculating the P(Survived| Xi = k)
            # Using the average of the log probabilites for missing data 
            surv_list = prob_attr_yes[att]

            if row_t.iloc[0][att] in surv_list:
                prob_test_survived += surv_list[row_t.iloc[0][att]]
                pi_s += math.exp(surv_list[row_t.iloc[0][att]])
            else:
                prob_test_survived += sum(surv_list.values()) / float(len(surv_list))
                pi_s += math.exp(sum(surv_list.values()) / float(len(surv_list)))
                
            # Calculating the P(Dead | Xi = k)
            dead_list = prob_attr_no[att]
            if row_t.iloc[0][att] in dead_list:
                prob_test_dead += dead_list[row_t.iloc[0][att]]
                pi_d += math.exp(dead_list[row_t.iloc[0][att]])
            else:
                prob_test_dead += sum(dead_list.values()) / float(len(dead_list))
                pi_d += math.exp(sum(dead_list.values()) / float(len(dead_list)))
                
        
        # Calculating the P(X)
        p_x = pi_s + pi_d
            
        # Calculating Sqared Loss
        if (row_t.iloc[0]["Survived"] == 1):
            each_pi = pi_s
        else:
            each_pi = pi_d
        
        pi = each_pi / p_x
        sl += ((1 - pi)**2)

        # 0-1 Loss
        if (pi < 0.5):
            loss += 1
            
        # Whichever one is greater is the final test label 
        if prob_test_survived > prob_test_dead:
            predictions.append(1) 
        else:
            predictions.append(0)
         
    loss = loss / float(len(test_data))    
    sl = sl / float(len(test_data))  
    return predictions, loss, sl   

    
        
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
    # apply_cross_validation(train_attr_data, train_label_data)
    
    # UNCOMMENT both line to do baseline analsis 
    # make_baseline_prediction(train_label_data, test_label_data)
    # sys.exit(0)
        
    # Running the model on the train and test set
    prob_survived, prob_dead, prob_attr_yes, prob_attr_no = train_NBC(train_attr_data, train_label_data)
    predictions, loss, squared_loss = predict_NBC(test_attr_data, test_label_data, prob_survived, prob_dead, prob_attr_yes, prob_attr_no) 
   
    # Calculating the test accuracy and zero-one loss
    accuracy = calc_accuracy(np.asarray(test_label_data), predictions)
    print("ZERO-ONE LOSS=%0.4f" % (loss))
    print("SQUARED LOSS=%0.4f  Test Accuracy=%0.4f" % (squared_loss, accuracy ))

    
    