# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 23:21:43 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
x = np.load("fashion_mnist_images.npy")
y = np.load("fashion_mnist_labels.npy")
d, n= x.shape #(784, 6000)
i = 0 #Index of the image to be visualized
#one column = one image

x = np.concatenate([np.ones([1, n]), x], axis=0)
x_train = x[:,:5000]
x_test = x[:,5000:]
y_train = y[:,:5000]
y_test = y[:,5000:]

print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("x_test: ",x_test.shape)
print("y_test: ",y_test.shape)

# phi funtion from Q3
def phi(t):
    return np.log(1 + np.exp(-t))

theta = np.zeros([d+1, 1]) # starts with a zero vector
lambd = 1 #Regularization constant
eps = 10e-6 #to break the loop

#iteration for updating weights
j = x_train.shape[1]*np.log(2)  
i = 0

while True:
    i+=1
    j_before = j # condition for breaking the loop
    g = 2 * lambd * theta
    j = lambd * np.dot(theta.T,theta)  #constant of J (not in sum)
    h = 2 * lambd * np.identity(d+1)
    #sum for every images (sum for the whole batch)
    for k in range(0, x_train.shape[1]):
        x_k = x_train[:,k].reshape(d+1,1)
        ytx = y[:,k]*np.dot(theta.T,x_k)
        j += phi(ytx)    #phi(np.exp(y[:,k]*np.dot(theta.T,x_k)))
        g -= y_train[:,k]*x_k*(1/(1+np.exp(ytx)))
        h += x_k.dot(x_k.T)*(y_train[:,k]**2)*(np.exp(ytx)/((1+np.exp(ytx))**2))
    
    theta -= np.dot(np.linalg.inv(h),g)
    
    if (np.abs(j-j_before)/j_before < eps) and i > 1 :
        break
    
y_pred = np.ones([y_test.shape[1],1])*-1 # because y is either (-1,1) according to Q3
y_pred[np.dot(x_test.T,theta)>=0] = 1
test_err = (np.sum(y_pred != y_test.T) / y_test.shape[1])*100 # ( number of misclassified data / number of test data ) * 100    

#confidence of each data
conf = abs(np.dot(x_test.T,theta))
#index of miss-classified data
miss_class_idx = np.where(y_pred != y_test.T)[0]

high_conf_sort = np.argsort(conf[miss_class_idx], axis = 0)[::-1]
high_conf_sort= np.squeeze(high_conf_sort, axis = 1)

imgs = x_test[1:,high_conf_sort] # slice from 1: to delete bias


fig, axes = plt.subplots(4, 5, figsize=(5, 5))
fig.subplots_adjust(hspace=.5, wspace=.001)

axes = axes.ravel()

for k in range(20):
    idx = high_conf_sort[k]
    if y_test[:, idx] == 1:
        true_lab = '1'
    else:
        true_lab = '-1'

    imgss = np.reshape(x_test[1:, idx], (int(np.sqrt(d)), int(np.sqrt(d))))
    axes[k].imshow(imgss)
    axes[k].axis('off')
    axes[k].set_title(true_lab)
    

plt.show()


print(f'test error: {test_err} % or {test_err/100}')
print(f'number of iteration: {i}')
print(f'value of objective function: {j[0][0]}')
