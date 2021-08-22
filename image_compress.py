import sys
import numpy as np
import seaborn as sns
import scipy.io as scyio
from functions import find_closest_centroid
from functions import compute_centroids
from functions import run_kmean_algo
from scipy.misc import imread
from numpy import double
from functions import kmeans_init_centroids
import matplotlib.pyplot as plt
import matplotlib as mpl

#Loading the sample data.
data = scyio.loadmat('ex7data2.mat')

#Here we are applying K-Means algorithm.
K = 3

# We have to initialize centroids.
initial_centroids = np.matrix([[3,3],[6,2],[8,5]])

#Initializing Data X to make clusters
X = np.matrix(data['X'])

print("Printing Closest Centroids....\n\n\n")
#Finding the nearest centroids.
idx = find_closest_centroid(X,initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(idx[0:3]);

# Now, we will calculate means.
print('\nComputing centroids means.\n\n')

# Compute means based on closest centroids found in the previous part.
centroids = compute_centroids(X,idx,K)

print('Centroids computed after initial finding of closest centroids: \n')
print(centroids)

#Now, we will apply K-means clustering algorithm.
# We will use same dataset for clustering.
max_iters = 10
#Runs K-means algorithm
# The 'true' at the end tells our function to plot the progress of K-Means.
centroids,idx = run_kmean_algo(X,initial_centroids,max_iters,True)
print('\nK-Means Done.\n\n')


# Applying K-means algorithm on Pixels.
print('\nRunning K-Means clustering on pixels from an image.\n\n')
#Loading my image.
img = plt.imread('universe.png')

# Deviding by 255 to Normalize
A = img/255

# Size of image
img_size = A.shape

# Reshape the image into Nx3 matrix where N = number of pixels
X = A.reshape(img_size[0]*img_size[1],3)

#print(X.shape)

# Running the K-means algorithm on this data
K = 16
max_iters = 10

# When using K-means, it is important to initialize the centroids randomly
initial_centroids = kmeans_init_centroids(X,K) 
#print(initial_centroids.shape)

# Run K-Means
centroids,idx = run_kmean_algo(X,initial_centroids,max_iters,False)

# Now we will apply image compression.
print('\nApplying K-Means to compress an image.\n\n')

# Find the closest cluster members
idx = find_closest_centroid(X,centroids)

#print(idx.shape)
#print(centroids.shape)
# Now, we have represented image X as in terms of the indices in idx
# We can recover the image from indices by mapping each pixel by centroid value.
#X_recovered = centroids[idx,:]
X_recovered = np.zeros(shape=(idx.shape[0],centroids.shape[1]))
#for i in range(0,idx.shape[0]):
#    X_recovered[i,:] = centroids[int(idx[i,0]),:]
id = []
for i in range(0,idx.shape[0]):
    id.append(int(idx[i,0]))
idx = id
X_recovered = centroids[idx,:]
#print(X_recovered.shape)

# Reshape the recovered image into proper dimensions.
X_recovered = X_recovered.reshape(img_size)

# Display the original image.
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,9))
ax1.imshow(img)
ax1.set_title('Original')
ax2.imshow(X_recovered*255)
ax2.set_title('Compressed, with 16 colors')
for ax in fig.axes:
    ax.axis('off')
plt.show()







