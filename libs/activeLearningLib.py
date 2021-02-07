from sklearn.cluster import KMeans
from libs.classifiersLib import classifiersLib
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
import numpy as np
import random

class activeLearningLib:

    # classifiersLibObject = classifiersLib()
   
    # Function that selects the samples at random and passes it to the label specialist.
    def randomActiveLearning(self, K, X, Y):
        # Convert to DataFrame
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        # Create DF to store the data selected at random
        auxDF = pd.DataFrame(columns = pd.concat([X, Y], axis = 1).columns.tolist())
        # And a DF with the data and labels concatenaded
        df = pd.concat([X, Y], axis = 1)
        # Select K samples and drop the separate lines to compose in the dataset labeled by the specialist
        for i in range(0, K):
            # Define the random seeds
            random.seed(200)
            # Select a random index within the size of the Dataframe.
            index = random.randrange(len(df.index))
            # Adds to the auxiliary Dataframe the index line selected at random.
            auxDF = auxDF.append(df.iloc[[index - 1]])
            # Removes the index line passed to the new Dataframe from the global Dataframe.
            df = df.drop(df.index[index - 1])
        # And here we update the pool of supervised data, dividing only the features of the classes.
        X = df.iloc[:, :len(df.columns) - 1]
        Y = df.iloc[:, -1:]
        return 0, auxDF.iloc[:, :-1].values, auxDF.iloc[:, -1:].values, X.values, Y.values

    # Function to get the root from clusters
    def getRootSamples(self, K, X, Y):
        # print("Selecionando as amostras raízes..")
        # print("")
        # Apply kmeans with K = ? and random_state 1 to get the same result everytime
        kmeansModel = KMeans(n_clusters = K, random_state = 200).fit(X)
        # Get the labels from the cluster
        kmeans_labels = kmeansModel.labels_
        # Get the distances from the centroids
        kmeans_distances = kmeansModel.transform(X)
        # Root Samples Index
        rootSamples = []
        # Loop through K samples to get the roots
        for i in range(0, K):
            aux = []
            for j in kmeans_distances:
                aux.append(j[i])
            rootSamples.append(aux.index(min(aux)))
            # print("min value C{} = {}".format(i, rootSamples[i]))
        # Pass selected samples and delete from data
        selected_X = X[rootSamples]
        selected_Y = Y[rootSamples]
        X = np.delete(X, rootSamples, axis = 0)
        Y = np.delete(Y, rootSamples, axis = 0)
        # print("")
        return rootSamples, selected_X, selected_Y, X, Y


    def getOrderedDistances(self, num_clusters, sup_X):
        # Apply kmeans with K = ? and random_state 1 to get the same result everytime
        kmeansModel = KMeans(n_clusters = num_clusters, random_state = 200).fit(sup_X)
        # Get the labels from the cluster
        kmeans_labels = kmeansModel.labels_
        # Get the distances from the centroids
        kmeans_distances = kmeansModel.transform(sup_X)
        # Reset index from sup_X
        sup_X = sup_X.reset_index()
        sup_X_aux = sup_X
        # Loop to go through the clusters and save the clusters distances into Series format, after
        # concatenate with the sup_X data
        for i in range(0, num_clusters):
            list_aux = []
            for j in kmeans_distances:
                list_aux.append(j[i])
            sup_X_aux = pd.concat([sup_X_aux, pd.Series(list_aux).rename('C' + str(i + 1))], axis = 1)
        # Go back the index to the sup_X
        sup_X_aux = sup_X_aux.set_index('index').rename_axis(index = None, columns = None)
        sup_X = sup_X.set_index('index').rename_axis(index = None, columns = None)
        # Leave just the centroids columns
        sup_X_aux = sup_X_aux.iloc[:, -(i + 1):]
        # Put the labels together
        sup_X_aux['labels'] = kmeans_labels
        # Split the clusters
        orderedDistancesByCluster = []
        for i in sup_X_aux['labels'].unique():
            orderedDistancesByCluster.append(sup_X_aux[sup_X_aux['labels'] == i].iloc[:, np.r_[i, len(sup_X_aux.columns) - 1]].sort_values(by = 'C' + str(i + 1)))
        return orderedDistancesByCluster

    # def trainClassifiers(self, labeledData, selectedClassifier):
    #     # Call the function to split into features and class
    #     X, Y = self.classifiersLibObject.splitFeaturesClass(labeledData)
    #     if selectedClassifier == "GaussianNB":
    #         gnb = GaussianNB()
    #         model = gnb.fit(X, Y)
    #     elif selectedClassifier == "LogisticRegression":
    #         logreg = LogisticRegression()
    #         model = logreg.fit(X, Y)
    #     elif selectedClassifier == "DecisionTree":
    #         dectree = DecisionTreeClassifier()
    #         model = dectree.fit(X, Y)
    #     elif selectedClassifier == "k-NN":
    #         knn = KNeighborsClassifier(n_neighbors = 2)
    #         model = knn.fit(X, Y)
    #     elif selectedClassifier == "LDA":
    #         lda = LinearDiscriminantAnalysis()
    #         model = lda.fit(X, Y)
    #     elif selectedClassifier == "SVM":
    #         svm = SVC()
    #         model = svm.fit(X, Y)
    #     elif selectedClassifier == "RandomForest":
    #         rf = RandomForestClassifier()
    #         model = rf.fit(X, Y)
    #     elif selectedClassifier == "NeuralNet":
    #         nnet = MLPClassifier()
    #         model = nnet.fit(X, Y)
    #     return model

    # def samplesByRDS(self, rdsLists, model, sup_X, truLabel_rdsLists, sup_Y, valueOfK):
    #     labelCorrected = 0
    #     ifTerminatedSamples = 0
    #     # Create DF to store the data selected at RDS
    #     auxDF = pd.DataFrame(columns = pd.concat([sup_X, sup_Y], axis = 1).columns.tolist())
    #     # And a DF with the data and labels concatenaded
    #     df = pd.concat([sup_X, sup_Y], axis = 1)
    #     # variable to control de cluster number to get the truth labels
    #     cluster_number = 0
    #     # go through the clusters in list
    #     for i in rdsLists:
    #         # for each list (cluster) get the samples to classify each one and take the different
    #         selected = False
    #         for j in i.index:
    #             # Get the true label of each cluster
    #             cluster_label = np.asarray(truLabel_rdsLists.values).flatten()[cluster_number]
    #             # if the predicted is different of the cluster true label
    #             if model.predict([sup_X.loc[j, :]]) != cluster_label:
    #                 if (df.loc[j, :].iloc[-1]) != cluster_label:
    #                     # Sum Label Corrected
    #                     labelCorrected += 1
    #                 print("Amostra com indice {} do Agrupamento {} selecionada por ter o rotulo {} dado pelo Classificador, diferente do TrueLabel {}"
    #                 .format(j, cluster_number + 1, model.predict([sup_X.loc[j, :]]), cluster_label))
    #                 print("")
    #                 # To control the selected samples per list (to get the last value if dont true)
    #                 selected = True
    #                 # append the selected samples to dataframe
    #                 auxDF = auxDF.append(df.loc[j, :])
    #                 # Removes the index line passed to the new Dataframe from the global Dataframe.
    #                 df = df.drop(index = [j])
    #                 # drop the values from list
    #                 i = i.drop(index = [j])
    #                 # And here we update the pool of supervised data, dividing only the features of the classes.
    #                 sup_X = df.iloc[:, :len(df.columns) - 1]
    #                 sup_Y = df.iloc[:, -1:]
    #                 break;
    #         if selected == False:
    #             try:
    #                 if (df.loc[j, :].iloc[-1]) != cluster_label:
    #                     # Sum Label Corrected
    #                     labelCorrected += 1
    #                 print("Amostra com indice {} selecionada por causa do classificador não ter identificado nenhuma amostra com rótulo diferente."
    #                     .format(i.index[-1]))
    #                 print("")
    #                 # append the selected samples to dataframe
    #                 auxDF = auxDF.append(df.loc[i.index[-1], :])
    #                 # Removes the index line passed to the new Dataframe from the global Dataframe.
    #                 df = df.drop(index = [i.index[-1]])
    #                 # drop the values from list
    #                 i = i.drop(index = [i.index[-1]])
    #                 # And here we update the pool of supervised data, dividing only the features of the classes.
    #                 sup_X = df.iloc[:, :len(df.columns) - 1]
    #                 sup_Y = df.iloc[:, -1:]
    #             except:
    #                 print("Lista do cluster número {} sem mais amostras.".format(cluster_number + 1))
    #                 ifTerminatedSamples+=1
    #         if ifTerminatedSamples == valueOfK:
    #             print("Não há mais amostras para serem rotuladas pelo especialista..")
    #             auxDF = 0
    #         rdsLists[cluster_number] = i
    #         cluster_number += 1
    #     return rdsLists, sup_X, sup_Y, auxDF, labelCorrected

    # def rds2(self, K, sup_X, sup_Y, first, rdsLists, labeledData, truLabel_rdsLists, rdsClassifier):
    #     # Create DF to store the data selected at RDS
    #     auxDF = pd.DataFrame(columns = pd.concat([sup_X, sup_Y], axis = 1).columns.tolist())
    #     print("Número de amostras no Conjunto Supervisionado: {}".format(len(sup_Y.values) - K))
    #     print("")
    #     # And a DF with the data and labels concatenaded
    #     df = pd.concat([sup_X, sup_Y], axis = 1)
    #     # first = the first time that we call rds and need to create the list's with ordered clusters
    #     if first:
    #         # function return ordered lists
    #         rdsLists = self.getOrderedDistances(K, sup_X)
    #         # for go through cluster's numbers
    #         for i in range(0, K):
    #             # append the "root samples" (first index of each list) to dataframe
    #             auxDF = auxDF.append(df.loc[rdsLists[i].index[0], :])
    #             # Removes the index line passed to the new Dataframe from the global Dataframe.
    #             df = df.drop(index = [rdsLists[i].index[0]])
    #             # drop the values from list
    #             rdsLists[i] = rdsLists[i].drop(index = [rdsLists[i].index[0]])
    #             # And here we update the pool of supervised data, dividing only the features of the classes.
    #             sup_X = df.iloc[:, :len(df.columns) - 1]
    #             sup_Y = df.iloc[:, -1:]
    #         # Get the true labels from the each cluster list
    #         truLabel_rdsLists = auxDF.iloc[:, -1:]
    #         return auxDF, sup_X, sup_Y, rdsLists, truLabel_rdsLists
    #     else:
    #         model = self.trainClassifiers(labeledData, rdsClassifier)
    #         rdsLists, sup_X, sup_Y, auxDF = self.selectSamples(rdsLists, model, sup_X, truLabel_rdsLists, sup_Y)
    #         return auxDF, sup_X, sup_Y, rdsLists, truLabel_rdsLists

    def unique_without_sorting(self, array):
        indexes = np.unique(array, return_index = True)[1]
        return [array[idx] for idx in sorted(indexes)]

    def get_bondary_edges(self, x, cluster_ids):
        knn_dist, knn_idx = self.get_knn(x)
        bedges_idx = np.empty((0, 2), int)
        bedges_dist = np.empty((0, 1), float)
        for i, knn_pair in enumerate(knn_idx):
            if cluster_ids[knn_pair[0]] != cluster_ids[knn_pair[1]] and \
                    not np.any(np.all(bedges_idx == np.flip(knn_pair, axis = 0), axis = 1)):
                bedges_idx = np.vstack((bedges_idx, knn_pair))
                bedges_dist = np.append(bedges_dist, knn_dist[i][1])
        return bedges_dist, bedges_idx

    def get_boundary_idx(self, x, cluster_ids, as_edges = True, order = None):
        bedges_dist, bedges_idx = self.get_bondary_edges(x, cluster_ids)
        if order != None:
            stack = np.column_stack((bedges_dist, bedges_idx))
            if order == 'desc':
                stack = stack[stack[:, 0].argsort()[::-1]]
            elif order == 'asc':
                stack = stack[stack[:, 0].argsort()]
            bedges_idx = stack[:, 1:].astype(int)
        idx = bedges_idx.flatten()
        if not as_edges:
            idx = self.unique_without_sorting(idx)
        return idx

    def get_mst_idx(self, x):
        knn_graph = self.get_knn(x, n_neighbors = len(x), return_graph = True)
        mst_array = minimum_spanning_tree(knn_graph).toarray()
        nonzero_indices = np.asarray(mst_array.nonzero())
        data_argsort = mst_array[mst_array.nonzero()].argsort()
        idx = nonzero_indices[:, data_argsort].flatten('F')
        return idx[::-1]

    def get_clusters_dict(self, cluster_ids):
        clusters_dict = dict()
        y_unique = np.unique(cluster_ids)
        for c in y_unique:
            clusters_dict[c] = np.argwhere(cluster_ids == c).flatten()
        return clusters_dict

    def get_rds_cluster_dict(self, x, cluster_ids, cluster_centers):
        cluster_dict = self.get_clusters_dict(cluster_ids)
        rds_cluster_dict = dict()
        for c in cluster_dict:
            cluster_samples = cluster_dict[c]
            if len(cluster_samples) <= 1:
                continue
            _, knn_idx = self.get_knn(x[cluster_samples], neighbors = cluster_centers[c:c + 1],
                                n_neighbors = len(cluster_samples))
            rds_cluster_dict[c] = cluster_samples[knn_idx].flatten()
        return rds_cluster_dict

    def get_knn(self, x, neighbors = None, n_neighbors = 2, return_graph = False):
        nbrs = NearestNeighbors(n_neighbors = n_neighbors, algorithm = 'auto')  # , metric='euclidean')
        nbrs.fit(x)
        knn = neighbors if neighbors is not None else x
        if not return_graph:
            knn_dist, knn_idx = nbrs.kneighbors(knn)
            return knn_dist, knn_idx
        else:
            return nbrs.kneighbors_graph(knn, mode='distance')

    def get_root_idx(self, x, cluster_centers):
        _, knn_idx = self.get_knn(x, neighbors = cluster_centers, n_neighbors = 1)
        idx = knn_idx.flatten()
        return idx

    def get_initial_data(self, cluster_centers, x, y):
        root_idx = self.get_root_idx(x, cluster_centers)
        n_clusters = len(cluster_centers)
        while len(np.unique(y[root_idx])) < 2:
            n_clusters += 1
            _, cluster_centers = self.cluster_data(x, n_clusters)
            root_idx = self.get_root_idx(x, cluster_centers)
        return root_idx, x[root_idx], y[root_idx]
        
    def cluster_data(self, x, n_clusters):
        kmeans = KMeans(n_clusters = int(n_clusters), random_state = 1).fit(x)
        cluster_ids = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        return cluster_ids, cluster_centers

    def get_samples(self, x, y, n_clusters = None, strategy = None):
        cluster_ids, cluster_centers = self.cluster_data(x, n_clusters)
        root_idx, x_initial, y_initial = self.get_initial_data(cluster_centers, x, y)
        x_pool = np.delete(x, root_idx, axis = 0)
        y_pool = np.delete(y, root_idx, axis = 0)
        cluster_ids = np.delete(cluster_ids, root_idx, axis = 0)
        organized_data = np.empty(0, int)
        if strategy == "RDS":
            organized_data = self.get_rds_cluster_dict(x_pool, cluster_ids, cluster_centers)
        elif strategy == "MST-BE":
            organized_data = self.get_boundary_idx(x_pool, cluster_ids, as_edges = False)
            x_pool, y_pool = x_pool[organized_data], y_pool[organized_data]
            organized_data = self.get_mst_idx(x_pool)
        return organized_data, root_idx, x_initial, y_initial, x_pool, y_pool