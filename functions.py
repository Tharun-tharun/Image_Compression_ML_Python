import numpy as np
import sklearn
import scipy.io as scyio
import matplotlib.pyplot as plt
from django.core.management import color

def find_closest_centroid(X,centroids):
    # Number of centroids taken
    K = centroids.shape[0]
    # Number of examples
    m = X.shape[0]
    #Initializing np matrix of zeros for idx
    idx = np.zeros(shape=(m,1))
    
    # Computing idx for each example.
    for i in range(0,m):
        arr = []
        for j in range(0,K):
            a = np.sqrt(np.sum(np.power(X[i][:]-centroids[j][:],2)))
            arr.append(a)
        idx[i,0] = arr.index(min(arr))
    return idx

def compute_centroids(X,idx,K):
    (m,n) = X.shape
    centroids = np.zeros(shape=(K,n))
    
    #Computing means
    for i in range(0,K):
        co = 0
        for j in range(0,m):
            if(idx[j,0]==i):
                co+=1
                centroids[i,:] = centroids[i,:] + X[j,:]
        if co!=0:
            centroids[i,:] = (1/co)*centroids[i,:]
    return centroids

def run_kmean_algo(X,initial_centroids,max_iters,plot_progress):
    # Initializing values.
    (m,n) = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(shape=(m,1))
    
    # Run K-Means
    for i in range(0,max_iters):
        # Output Progress
        print("K-means iteration %d"%i,end="")
        print("/%d"%max_iters)
        
        # For each example in X, assign it to closest centroid.
        idx = find_closest_centroid(X, centroids)
        
        #Plot Progress.
        if plot_progress:
            plot_progress_kmeans(X,centroids,previous_centroids,idx,K,i)
            previous_centroids = centroids
        
        # Given Membership, compute new centroids.
        centroids = compute_centroids(X, idx, K)
    return centroids,idx

def plot_data_points(X,idx,K,centroids):
    # Initializing three sets
    (m,n) = X.shape
    X0x = []
    X0y = []
    X1x = []
    X1y = []
    X2x = []
    X2y = []
    
    # Distributing values
    for i in range(0,m):
        if idx[i,0]==0:
            X0x.append(X[i,0])
            X0y.append(X[i,1])
        elif idx[i,0]==1:
            X1x.append(X[i,0])
            X1y.append(X[i,1])
        elif idx[i,0]==2:
            X2x.append(X[i,0])
            X2y.append(X[i,1])
    
    #Distributing centroids
    Cx = []
    Cy = []
    for i in range(0,centroids.shape[0]):
        Cx.append(centroids[i,0])
        Cy.append(centroids[i,1])
    
    # Plotiing on Scatter plot
    plt.scatter(X0x, X0y, marker='o',color='red')
    plt.scatter(X1x,X1y,marker='o',color='blue')
    plt.scatter(X2x,X2y,marker='o',color='green')
    plt.scatter(Cx,Cy,marker='X',color='black')
    plt.show()
    #plt.savefig('scatter_cluster.png')

def plot_progress_kmeans(X,centroids,previous,idx,K,i):
    #Plot Examples
    plot_data_points(X,idx,K,centroids)
    
def kmeans_init_centroids(X,K):
    return X[np.random.choice(X.shape[0], K)]
    
            
            
            
            
            
            
            
            
    