# Kernel Logistic Regression
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.metrics import accuracy_score

#Linear Logistic regression for performance comparison
def linear_logistic_regression(x_train, y_train, x_test, y_test, step_size, reg_strength, num_iters):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1/reg_strength)
    clf.fit(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    return test_acc

def kernel_logistic_regression(x_train, y_train, x_test, y_test, step_size, reg_strength, num_iters, kernel_parameter):
    """
    x_train - (n_train, d)
    y_train - (n_train,)
    x_test - (n_test, d)
    y_test - (n_test,)
    step_size: gamma in problem description
    reg_strength: lambda in problem description
    num_iters: how many iterations of gradient descent to perform

    Implement KLR with the Gaussian Kernel.
    The only allowed sklearn usage is the rbf_kernel, which has already been imported.
    """
    #train
    n = len(x_train)
    a = np.zeros(n).T
    b = 0
    
    kx=rbf_kernel(x_train)
    
    for e in range(200):
        
        b_k1=1/n*(np.sum(y_train/(1+np.exp(y_train*(np.dot(a,kx)+b)))))
        a = a*(1-2*reg_strength*step_size)+step_size*y_train/(n*(1+np.exp(y_train*np.dot(a,kx)+b)))
        b=b+step_size*b_k1
   
    #test
    k_test=rbf_kernel(x_train, x_test, gamma = kernel_parameter)
    y_pred=np.sign(np.dot(a, k_test)+np.repeat(b,len(x_test)))
    print(y_pred)
    score = accuracy_score(y_test, y_pred)
    return score

x_train = np.load("x_train.npy")    # shape (n_train, d)
x_test = np.load("x_test.npy")      # shape (n_test, d)

y_train = np.load("y_train.npy")    # shape (n_train,)
y_test = np.load("y_test.npy")        # shape (n_test,)

linear_acc = linear_logistic_regression(x_train, y_train, x_test, y_test, 1.0, 0.001, 200)
print("Linear LR accuracy:", linear_acc)

klr_acc = kernel_logistic_regression(x_train, y_train, x_test, y_test, 5.0, 0.001,200, 0.1)
print("Kernel LR accuracy:", klr_acc)



