'''
    files=glob.glob(faces_folder + "s*/*.pgm")   # list of filenames   import glob
    img=np.array(PGM.open(faces_folder + "s" + str(person+1) + "/" + str(faces+1) + ".pgm"))
    print(img.shape)
    print(img)
    plt.imshow(img, cmap='hot')
    print(np.asarray(Xt).shape)
    plt.imshow(np.asarray(Xt)[10], cmap='hot')
'''

# not use
def read_pgm_file (file_name):
    """Return a raster of integers from a PGM as a list of lists """
    assert file_name.readline() == 'P5\n'
    (width, height) = [int(i) for i in file_name.readline().split()]
    depth = int(file_name.readline())
    assert depth <= 255
    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(file_name.read(1)))
        raster.append(row)
    return raster


%%time
fig = plt.figure(figsize=(faces_count_verif*X_verif_orig.shape[2]/32, person_count*X_verif_orig.shape[1]/32)) 
for i in range(X_verif_orig.shape[0]):
    fig.add_subplot(person_count, faces_count_verif, i+1)
    plt.imshow(X_verif_orig[i], cmap='hot')
plt.show()




# http://seat.massey.ac.nz/personal/s.r.marsland/Code/Ch6/pca.py

# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# An algorithm to compute PCA. Not as fast as the NumPy implementation
import numpy as np

def pca(data,nRedDim=0,normalise=1):
    
    # Centre data
    m = np.mean(data,axis=0)
    data -= m

    # Covariance matrix
    C = np.cov(np.transpose(data))

    # Compute eigenvalues and sort into descending order
    evals,evecs = np.linalg.eig(C) 
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]

    if nRedDim>0:
        evecs = evecs[:,:nRedDim]
    
    if normalise:
        for i in range(np.shape(evecs)[1]):
            evecs[:,i] / np.linalg.norm(evecs[:,i]) * np.sqrt(evals[i])

    # Produce the new data matrix
    x = np.dot(np.transpose(evecs),np.transpose(data))
    # Compute the original data again
    y=np.transpose(np.dot(evecs,x))+m
    return x,y,evals,evecs




# ??? COMPLEX DATA
A = np.array([[1,2,1,3,5,3], 
              [2,3,1,5,8,4],
              [3,4,7,8,5,5]])
X = norm_basic(A)  #X_train
print(X.shape)
print(X)
print("---------")

C = np.dot(X.T,X)  # build covariance matrix   (C = np.cov(X.T))
print(C.shape)

val, vec, v = np.linalg.svd(C.T)  # calc eigenvalues, eigenvectors  (val, vec = np.linalg.eig(C)) (vec, val, v = np.linalg.svd(C.T))
print(val)
print(vec)
print("---------")
val, vec = np.linalg.eig(C)
print(val)
print(vec)




A = X_train[:12,:15]
X = norm_mean(A)  
print(X)

# build covariance matrix
C = np.dot(X.T,X)   # np.cov(X.T)  # np.dot(X.T,X)    
print("--------- covariance matrix")
print(C)

 # calc eigenvalues, eigenvectors 
val, vec = np.linalg.eig(C)     # vec, val, v = np.linalg.svd(C)
print("--------- eigenvalues, eigenvectors")
print(val)
print(vec)

# desc sort eigen* by eigenvalues 
indices = np.argsort(val)
indices = indices[::-1]
vec = vec[:,indices]
val = val[indices]

pca_energy = 0.95
energy = 0.0
k = 0;
while (energy < pca_energy) and (k < val.shape[0]):  
    k+=1
    energy = np.sum(val[:k])/np.sum(val)
    print(energy)
vec = vec[:,:k]    

print(X.shape)
X = np.dot(X,vec)
print(X.shape)




