from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

class activeSelectionLib:

    def randomSelection(self, X_train, Y_train):
        # Concatenate the data and labels
        df = pd.concat([X_train, Y_train], axis = 1)
        # Split randomized, with a seed of 200
        supervised = df.sample(frac = 0.5, random_state = 1)
        unsupervised = df.drop(supervised.index)
        return supervised.iloc[:, :-1], unsupervised.iloc[:, :-1], supervised.iloc[:,-1], unsupervised.iloc[:,-1]

    def activeNeighbors(self, X_train, Y_train, numClusters, numNeighbors):
        # Concatenate the data and labels
        trainingData = pd.concat([X_train, Y_train], axis = 1)
        # Create variables sup_X, sup_Y, unsup_X, unsup_Y
        sup_X = pd.DataFrame(columns = trainingData.columns.tolist())
        unsup_X = pd.DataFrame(columns = trainingData.columns.tolist())
        sup_Y = pd.DataFrame(columns = trainingData.columns.tolist())
        unsup_Y = pd.DataFrame(columns = trainingData.columns.tolist())
        # Apply kmeans with K = ? and random_state 1 to get the same result everytime
        kmeans = KMeans(n_clusters = numClusters, random_state = 1).fit(X_train)
        # Get the labels from the cluster
        Y_kmeans = kmeans.labels_
        # Train model to 10 x K Value nearest neighbors in the X_train
        knearest = NearestNeighbors(numNeighbors).fit(X_train)
        # Loop to go through all rows of the X_train
        for i in range(0, len(X_train)):
            # This boolean is to control the distribution of samples
            founded = False
            # This method returns the K nearest neighbors by index of the X_train from a passed sample
            # The parameter "return_distance = False" is to return the index of the sample not the distance between them
            X_knearest_train = knearest.kneighbors([X_train.iloc[i]], return_distance = False)
            # See that the first closest sample is the one consulted.
            for j in X_knearest_train[0][1:]:
                if Y_kmeans[i] != Y_kmeans[j]:
                    sup_X = sup_X.append(trainingData.iloc[i])
                    sup_Y = sup_Y.append(trainingData.iloc[i])
                    founded = True
                    break
            if founded == False:
                unsup_X = unsup_X.append(trainingData.iloc[i])
                unsup_Y = unsup_Y.append(trainingData.iloc[i])
        # Return results
        return sup_X.iloc[:, :-1], unsup_X.iloc[:, :-1], sup_Y.iloc[:,-1], unsup_Y.iloc[:,-1]