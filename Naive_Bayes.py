# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 23:01:18 2021

@author: User
"""
import numpy as np
import math

train_x = np.load(r'C:\Users\User\Desktop\EECS 545\HW2\hw2p2_data\hw2p2_train_x.npy')
train_y = np.load(r'C:\Users\User\Desktop\EECS 545\HW2\hw2p2_data\hw2p2_train_y.npy')
test_x = np.load(r'C:\Users\User\Desktop\EECS 545\HW2\hw2p2_data\hw2p2_test_x.npy')
test_y = np.load(r'C:\Users\User\Desktop\EECS 545\HW2\hw2p2_data\hw2p2_test_y.npy')
#train data 1192, 1000 words


#(c)
# c-(2) calculate prior (pi_k)
class_0_data = train_x[np.where(train_y == 0)] #data that belogs to class 0
class_1_data = train_x[np.where(train_y == 1)] #data that belogs to class 1

prior_0 = len(class_0_data) / len(train_x)
prior_1 = len(class_1_data) / len(train_x)

log_prior_0 = np.log(prior_0)
log_prior_1 = np.log(prior_1)

# c-(1) likelihood (p_kj)
# frequency of 1 in class0
# +1 on numerator / +1000 on denominator to avoid log(0)
likelihood_by_words0 = (np.count_nonzero(class_0_data, axis = 0)+1)/(len(class_0_data)+1000) #(1192,) => freq. for each words
# frequency of 1 in class1
likelihood_by_words1 = (np.count_nonzero(class_1_data, axis = 0)+1)/(len(class_1_data)+1000)

# make likelihood table which contains likelihood for each cases
# likelihood table for the value 1s, class 0, class1
likelihood_yes = np.vstack((likelihood_by_words0,likelihood_by_words1)) #when frequency >0
log_likelihood_yes = np.log(likelihood_yes) 
likelihood_no = 1-likelihood_yes   #when frequency = 0, because the likelihood table is the case when frequency > 0
log_likelihood_no = np.log(likelihood_no) 



# (d)
# testing, test error

data = test_x[0]
np.sum(log_likelihood_yes[0,np.where(data>0)])+np.sum(log_likelihood_no[0,np.where(data==0)])

prob_list_0=[]
prob_list_1=[]
label_list=[]
for d in range(0, len(test_x)):
    data = test_x[d]
    prob_0 = np.sum(log_likelihood_yes[0,np.where(data>0)])+np.sum(log_likelihood_no[0,np.where(data==0)])+log_prior_0 #prob for class 1
    prob_1 = np.sum(log_likelihood_yes[1,np.where(data>0)])+np.sum(log_likelihood_no[1,np.where(data==0)])+log_prior_1 #prob for class 2
    
    prob_list_0.append(prob_0)
    prob_list_1.append(prob_1)
    
    if prob_0 > prob_1:
        label = 0
    else:
        label = 1
    
    label_list.append(label)


test_err = np.sum(np.abs(label_list - test_y)) / len(test_y)
print(f"test error : {test_err} or {test_err*100}%")

#(e) majority prediction
if log_prior_0 > log_prior_1:
    y_maj_pred = np.zeros([test_y.shape[0]])
else:
    y_maj_pred = np.ones([test_y.shape[0]])
    
majority_err = np.sum(abs(test_y - y_maj_pred)) / test_y.shape[0]
print(f"majority prediction error : {majority_err} or {majority_err*100}%")
