import numpy as np
import matplotlib.pyplot as plt

#load data
X = np.load("./fairface.npy")

#image visualiazation
fig = plt.figure(figsize=(10, 5))
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    idx = np.random.randint(0, len(X) - 1)
    ax.imshow(X[idx], cmap=plt.get_cmap("gray"))

n,w,h=X.shape
d=w*h

#reshape 2D image to column vectors
#one column = one image data
data=np.reshape(X,[w*h,n])

S=np.cov(data) #sample cov
V,D=np.linalg.eig(S) #eigenvalue, eigenvector

Vsort = np.sort(V,axis = 0)
Vsort[:] = Vsort[::-1]
VsortInd = np.argsort(V,axis =0)
VsortInd[:] = VsortInd[::-1]
D = D[:,VsortInd]
V = np.copy(Vsort)
print(V)

plt.semilogy(V)
plt.show()

eigval = np.real(V)
D = np.real(D)

print(data.shape, X.shape, D.shape,V.shape)

# evaluate the number of principal components needed to represent 95% TV.
eigsum = np.sum(eigval)
ksum = 0
for i in range(0,d):
    ksum = ksum + eigval[i]
    tv = ksum/eigsum
    if tv > 0.95:
        k95 = i+1
        break
print ("TV achieved with {} principle components".format(k95))
print ("Reduction in dimension: {}".format(100*(1.0-(k95)/float(d+1))))

# evaluate the number of principal components needed to represent 99% TV.
eigsum = np.sum(eigval)
ksum = 0
for i in range(0,d):
	ksum = ksum + eigval[i]
	tv = ksum/eigsum
	if tv > 0.99:
		k99 = i+1
		break

print ("TV achieved with {} principle components".format(k99))
print ("Reduction in dimension: {}".format(100*(1.0-k99/float(d+1))))

# show 0th through 19th principal eigenvectors
eig0 = np.reshape(np.mean(data,axis =1), [w,h])

f, axarr = plt.subplots(4, 5)
for i in range(0,4):
	for j in range(0,5):
		if i == 0 and j ==0:
			axarr[i, j].imshow(eig0, cmap=plt.get_cmap('gray'))
		else:
			px = np.reshape(D[:,i*5+j-1],[dx,dy])
			axarr[i, j].imshow(px, cmap=plt.get_cmap('gray'))		
plt.show()