
########################## 1. Common Imports ##########################
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math
import csv

########################## 2. Data Preparation ##########################
# 2.1 Data Visualization

#data = pd.read_csv("data.csv")
#label = pd.read_csv("label.csv")

#print("Data Visualization")
#print("data.shape\t", data.shape)
#print("label.shape\t", label.shape)
#print("data.describe\n", data.describe())
#print("label.describe\n", label.describe())

# use below codes to obtain histogram in Jupyter
# data.hist(bins = 10, figsize = (20, 20))
# label.hist(bins = 50, figsize = (10, 5))

# 2.2 Read data
def get_label(filePath):
    t = []
    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    y = np.array(t).reshape((-1, 1))
    return y

def get_data(filePath):    
    dataMatrix = [] 
    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   

    dataMatrix = np.delete(dataMatrix, [5,6,7,8,9,45], axis=1)    

    return dataMatrix

# 2.3 Split Dataset
def split_data(x, y):
	len = x.shape[0]
	train = (int)(len * 0.8)
	test = (int)(len * 0.9)

	x_train = x[:train,:]
	x_validate = x[train:test,:]
	x_test = x[test:,:]

	y_train = y[:train,:]
	y_validate = y[train:test,:]
	y_test = y[test:,:]

	return x_train, x_test, x_validate, y_train, y_test, y_validate


########################## 3. K-Means Cluster ##########################
# 3.1 init centroids for k-means
def init_k_means_centroids(X, k):
    init_centroids = np.zeros([k, X.shape[1]])
      
    np.random.seed(42) # make results repeatable to be able to debug
    while(True):
        X_permute = np.random.permutation(X) # shuffle x
        init_centroids = X_permute[0:k, :]
        unique_centroids = (np.unique(init_centroids, axis = 0)).shape[0]
        if(unique_centroids== k): # The initial centroids should all be unique.
            break
    
    return init_centroids

# 3.2 find memberships
# groups holds the cluster number each element in X belongs to
def find_memberships(X, centroids):
    k = centroids.shape[0]
    m = X.shape[0]

    distances = np.zeros([m,k])   
    for i in range(k):
        diff = centroids[i] - X
        diff_sqr = diff ** 2
        diff_sum = np.sum(diff_sqr, axis = 1)
        distances[:,i] = diff_sum
        
    memberships = np.argmin(distances, axis = 1)
    return memberships

# 3.3 computer centroids
def compute_centroids(X, M, init_centroids, max_iters):
        
    centroids = init_centroids
    
    for loop in range(max_iters):
        prev_centroids = centroids.copy()
        
        memberships = find_memberships(X, centroids)
        index = []
        for i in range(M):
            index_one_cluster = np.where(memberships == i)
            index.append(index_one_cluster)
            
        for i in range(0, M):
            if(X[index[i]].size > 0): centroids[i,:] = np.mean(X[index[i]], axis = 0)
                
        if(np.array_equal(centroids,prev_centroids)):
            print("    Centroids converge after iterations: ", loop)
            cluster_sizes_csv(memberships, M, "cluster_distribution.csv")
            break
            
    return centroids, memberships

# 3.4 write cluster distribution to csv file
def cluster_sizes_csv(memberships, M, filename):
    clusters_size = []
    for i in range(M):
        index_one_cluster = np.where(memberships == i)
        clusters_size.append(memberships[index_one_cluster].size)
    pd.DataFrame(clusters_size).to_csv(filename, index = False)

# 3.5 do k-means
def k_means(X, M, max_iters):
    has_nan = True
    
    # check if centroids has nan, if does, do k-means again
    while(has_nan): 
        init_centroids = init_k_means_centroids(X, M)
        centroids, memberships = compute_centroids(X, M, init_centroids, max_iters)
        has_nan = np.isnan(centroids).any() 
    return centroids, memberships


########################## 4. Big Sigma ##########################
# Compute sigma based on each cluster
def get_sigma(X, M, memberships):

    N = X.shape[1]  
    sigma = np.zeros([M, N, N])
    
    for i in range(M):
        sigma_one_cluster = np.zeros([N, N]) 
        X_one_cluster = X[np.where(memberships == i)]
        for j in range(N):
            var = np.var(X_one_cluster[:,j])
            sigma_one_cluster[j][j] = var
        sigma[i] = sigma_one_cluster
    
    return sigma


########################## 5. Phi Matrix ##########################
# Gaussian radial basis function
def gauss_activation(X, Mu, Sigma):
    A = X - Mu
    A_t = A.T
    Sigma_inv = np.linalg.inv(Sigma)
    power = -0.5 * np.dot(np.dot(A, Sigma_inv),A_t)
    return math.exp(power)

# Compute pgi matrix
def get_phi_matrix(X, M, Mu, Sigma):
    N = X.shape[0]
    phi_matrix = np.zeros([N, M])
    for i in range(N):
        for j in range(M):
            phi_matrix[i][j] = gauss_activation(X[i], Mu[j], Sigma[j])
    return phi_matrix


########################## 6. Closed Form ##########################
# Implement closed-form solution
def w_closed_form(y, phi, Lambda):
    phi_t = phi.T
    I = np.identity(phi.shape[1])
    I[0,0] = 0 
    w = np.dot(np.linalg.inv(np.add(np.dot(phi_t,phi),Lambda*I)),np.dot(phi_t,y))
    return w

########################## 7. SGD Solution ##########################
# Divide data into batches according to batch_size
def next_batch(X, y, phi_matrix, batch_size):
    n = int(X.shape[0] / batch_size) * batch_size
    for i in np.arange(0, n, batch_size):
        yield (X[i:i + batch_size, :], y[i:i + batch_size, :], phi_matrix[i:i + batch_size, :])

# Implement SGD solution
def w_sgd(X, y, M, phi_matrix, learning_rate, Lambda, batch_size, epochs, early_stopping):
    W = np.random.normal(0, 10, size=(M,1))
    lossHistory = []
    for epoch in range(epochs):
        epochLoss = []
        for (batchX, batchY, phi) in next_batch(X, y, phi_matrix, batch_size): 
            
            W_pre = W
            preds = np.dot(phi, W)
            error = preds - batchY
            loss = np.sum(error ** 2, axis = 1)
            gradient = np.dot(phi.T, error)
                
            delta = -learning_rate * (gradient + Lambda * W_pre) 
            W += delta 
            
            epochLoss.append(np.sqrt(np.sum(loss)/batch_size))
        
        lossHistory.append(np.average(epochLoss))
        print("Epoch " + str(epoch) + "/" + str(epochs))
        print(str(batch_size) + "/" + str(batch_size) + "[=========================]" + " - loss: "+ str(np.average(epochLoss)))
        
          
        # add early stopping
        num = epochs
        if(epoch > 100):
            var_latest_epochs = lossHistory[-100:]
            
            if np.var(var_latest_epochs) < early_stopping:
                num = epoch + 1
                print("\n    Early stop at epoch: ", num)
                break
        
    # visualization   
    fig = plt.figure()
    plt.plot(range(num), lossHistory)
    fig.suptitle("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()
    
    #return W, lossHistory
    return W

########################## 8. Evaluation ############################
def evaluate(pred,y):
    sum = 0.0
    N = y.shape[0]
    for i in range (0, N):
        sum = sum + math.pow((y[i] - pred[i]),2)
    return math.sqrt(sum/N)

########################## Happy Ending ^0^ ##########################

