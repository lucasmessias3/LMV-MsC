from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from libs.utils import utils
import subprocess
import os
import pandas as pd
import numpy as np
import time

class classifiersLib:

    # Building the object's useds in this code
    utilsObject = utils()

    def splitFeaturesClass(self, labeledData):
        # Split data into features and labels
        X = labeledData.iloc[:, :len(labeledData.columns) - 1]
        Y = labeledData.iloc[:, -1:]
        # Change type Object to Integer
        Y = Y.astype('int')
        return X, Y

    # Gaussian Naive Bayes
    def gaussianNB(self, labeledData, X_test, Y_test):
        # Call the function to split into features and class
        X, Y = self.splitFeaturesClass(labeledData)
        # Define classifier and create model
        gnb = GaussianNB()
        model_gnb = gnb.fit(X, Y)
        # Make predictions and return/print the results
        aux = gnb.predict(X_test)
        print('Gaussian Naive Bayes:\nAcuracia: {}     F1-Score: {}     Precision: {}     Recall: {}'
        .format(accuracy_score(Y_test, aux).round(2), f1_score(Y_test, aux, average = 'macro').round(2),
            precision_score(Y_test, aux, average = 'macro').round(2), recall_score(Y_test, aux, average = 'macro').round(2)))
        # Print the confusion Matrix
        cm = confusion_matrix(Y_test, aux)
        print('Matriz de Confusão: ')
        print(cm)
        return accuracy_score(Y_test, aux), f1_score(Y_test, aux, average = 'macro'), precision_score(Y_test, aux, average = 'macro'), recall_score(Y_test, aux, average = 'macro')

    # Logistic Regression
    def logisticRegression(self, labeledData, X_test, Y_test):
        # Call the function to split into features and class
        X, Y = self.splitFeaturesClass(labeledData)
        # Define classifier and create model
        logreg = LogisticRegression()
        model_logreg = logreg.fit(X, Y)
        # Make predictions and return/print the results
        aux = logreg.predict(X_test)
        print('Logistic Regression:\nAcuracia: {}     F1-Score: {}     Precision: {}     Recall: {}'
        .format(accuracy_score(Y_test, aux).round(2), f1_score(Y_test, aux, average = 'macro').round(2),
            precision_score(Y_test, aux, average = 'macro').round(2), recall_score(Y_test, aux, average = 'macro').round(2)))
        # Print the confusion Matrix
        cm = confusion_matrix(Y_test, aux)
        print('Matriz de Confusão: ')
        print(cm)
        return accuracy_score(Y_test, aux), f1_score(Y_test, aux, average = 'macro'), precision_score(Y_test, aux, average = 'macro'), recall_score(Y_test, aux, average = 'macro')

    # Decision Tree
    def decisionTree(self, labeledData, X_test, Y_test):
        # Call the function to split into features and class
        X, Y = self.splitFeaturesClass(labeledData)
        # Define classifier and create model
        dectree = DecisionTreeClassifier()
        model_dectree = dectree.fit(X, Y)
        # Make predictions and return/print the results
        aux = dectree.predict(X_test)
        print('Decision Tree:\nAcuracia: {}     F1-Score: {}     Precision: {}     Recall: {}'
        .format(accuracy_score(Y_test, aux).round(2), f1_score(Y_test, aux, average = 'macro').round(2),
            precision_score(Y_test, aux, average = 'macro').round(2), recall_score(Y_test, aux, average = 'macro').round(2)))
        # Print the confusion Matrix
        cm = confusion_matrix(Y_test, aux)
        print('Matriz de Confusão: ')
        print(cm)
        return accuracy_score(Y_test, aux), f1_score(Y_test, aux, average = 'macro'), precision_score(Y_test, aux, average = 'macro'), recall_score(Y_test, aux, average = 'macro')

    # K-Nearest Neighbors
    def knn(self, labeledData, X_test, Y_test):
        # Call the function to split into features and class
        X, Y = self.splitFeaturesClass(labeledData)
        # Define classifier and create model
        knn = KNeighborsClassifier(n_neighbors = 2)
        model_knn = knn.fit(X, Y)
        # Make predictions and return/print the results
        aux = knn.predict(X_test)
        print('K-Nearest Neighbors:\nAcuracia: {}     F1-Score: {}     Precision: {}     Recall: {}'
        .format(accuracy_score(Y_test, aux).round(2), f1_score(Y_test, aux, average = 'macro').round(2),
            precision_score(Y_test, aux, average = 'macro').round(2), recall_score(Y_test, aux, average = 'macro').round(2)))
        # Print the confusion Matrix
        cm = confusion_matrix(Y_test, aux)
        print('Matriz de Confusão: ')
        print(cm)
        return accuracy_score(Y_test, aux), f1_score(Y_test, aux, average = 'macro'), precision_score(Y_test, aux, average = 'macro'), recall_score(Y_test, aux, average = 'macro')

    # Linear Discriminant Analysis
    def LinearDiscriminantAnalysis(self, labeledData, X_test, Y_test):
        # Call the function to split into features and class
        X, Y = self.splitFeaturesClass(labeledData)
        # Define classifier and create model
        lda = LinearDiscriminantAnalysis()
        model_lda = lda.fit(X, Y)
        # Make predictions and return/print the results
        aux = lda.predict(X_test)
        print('Linear Discriminant Analysis:\nAcuracia: {}     F1-Score: {}     Precision: {}     Recall: {}'
        .format(accuracy_score(Y_test, aux).round(2), f1_score(Y_test, aux, average = 'macro').round(2),
            precision_score(Y_test, aux, average = 'macro').round(2), recall_score(Y_test, aux, average = 'macro').round(2)))
        # Print the confusion Matrix
        cm = confusion_matrix(Y_test, aux)
        print('Matriz de Confusão: ')
        print(cm)
        return accuracy_score(Y_test, aux), f1_score(Y_test, aux, average = 'macro'), precision_score(Y_test, aux, average = 'macro'), recall_score(Y_test, aux, average = 'macro')

    # Support Vector Machine
    def supportVectorMachine(self, labeledData, X_test, Y_test):
        # Call the function to split into features and class
        X, Y = self.splitFeaturesClass(labeledData)
        # Define classifier and create model
        svm = SVC()
        model_svm = svm.fit(X, Y)
        # Make predictions and return/print the results
        aux = svm.predict(X_test)
        print('Support Vector Machine:\nAcuracia: {}     F1-Score: {}     Precision: {}     Recall: {}'
        .format(accuracy_score(Y_test, aux).round(2), f1_score(Y_test, aux, average = 'macro').round(2),
            precision_score(Y_test, aux, average = 'macro').round(2), recall_score(Y_test, aux, average = 'macro').round(2)))
        # Print the confusion Matrix
        cm = confusion_matrix(Y_test, aux)
        print('Matriz de Confusão: ')
        print(cm)
        return accuracy_score(Y_test, aux), f1_score(Y_test, aux, average = 'macro'), precision_score(Y_test, aux, average = 'macro'), recall_score(Y_test, aux, average = 'macro')

    # Random Forest
    def RandomForest(self, labeledData, X_test, Y_test):
        # Call the function to split into features and class
        X, Y = self.splitFeaturesClass(labeledData)
        # Define classifier and create model
        rf = RandomForestClassifier()
        model_rf = rf.fit(X, Y)
        # Make predictions and return/print the results
        aux = rf.predict(X_test)
        print('Random Forest:\nAcuracia: {}     F1-Score: {}     Precision: {}     Recall: {}'
        .format(accuracy_score(Y_test, aux).round(2), f1_score(Y_test, aux, average = 'macro').round(2),
            precision_score(Y_test, aux, average = 'macro').round(2), recall_score(Y_test, aux, average = 'macro').round(2)))
        # Print the confusion Matrix
        cm = confusion_matrix(Y_test, aux)
        print('Matriz de Confusão: ')
        print(cm)
        return accuracy_score(Y_test, aux), f1_score(Y_test, aux, average = 'macro'), precision_score(Y_test, aux, average = 'macro'), recall_score(Y_test, aux, average = 'macro')

    # Neural Net
    def neuralNet(self, labeledData, X_test, Y_test):
        print(labeledData.head(10))
        # Call the function to split into features and class
        X, Y = self.splitFeaturesClass(labeledData)
        # Define classifier and create model
        nnet = MLPClassifier()
        model_nnet = nnet.fit(X, Y)
        # Make predictions and return/print the results
        aux = nnet.predict(X_test)
        print('Neural Net (MLP):\nAcuracia: {}     F1-Score: {}     Precision: {}     Recall: {}'
        .format(accuracy_score(Y_test, aux).round(2), f1_score(Y_test, aux, average = 'macro').round(2),
            precision_score(Y_test, aux, average = 'macro').round(2), recall_score(Y_test, aux, average = 'macro').round(2)))
        # Print the confusion Matrix
        cm = confusion_matrix(Y_test, aux)
        print('Matriz de Confusão: ')
        print(cm)
        return accuracy_score(Y_test, aux), f1_score(Y_test, aux, average = 'macro'), precision_score(Y_test, aux, average = 'macro'), recall_score(Y_test, aux, average = 'macro')

    # Optimum Path-Forest
    def opf(self, labeledData_SS, X_test, Y_test):
        print(labeledData_SS.head(10))
        # Save labeledData in OPF format
        self.utilsObject.writeOPF(labeledData_SS, "training")
        # Join the two DataFrames to get the Test Data and save in OPF
        testData = pd.concat([X_test, Y_test], axis = 1)
        self.utilsObject.writeOPF(testData, "testing")
        # Run the bash file to execute OPF
        FNULL = open(os.devnull, 'w')
        subprocess.call(['/home/messias/Dropbox/GitHub/MsC-Project/framework/opf/./classifier.sh'], stdout = FNULL, stderr = subprocess.STDOUT)
        # Get the result from OPF
        aux = open('/home/messias/Dropbox/GitHub/MsC-Project/framework/opf/predict.txt', 'r').read().split('\n')[:-1]
        # Pass OPF result to Integer
        aux = [int(i) for i in aux]
        # Counts the unique samples of each class from the training data
        unique, counts = np.unique(aux, return_counts = True)
        for i in zip(unique, counts):
            print("      Número de amostras da Classe {} : {}".format(i[0], i[1]))
        print('OPF:\nAcuracia: {}     F1-Score: {}     Precision: {}     Recall: {}'
        .format(accuracy_score(Y_test, aux).round(2), f1_score(Y_test, aux, average = 'macro').round(2),
            precision_score(Y_test, aux, average = 'macro').round(2), recall_score(Y_test, aux, average = 'macro').round(2)))
        # Print the confusion Matrix
        cm = confusion_matrix(Y_test, aux)
        print('Matriz de Confusão: ')
        print(cm)
        return accuracy_score(Y_test, aux), f1_score(Y_test, aux, average = 'macro'), precision_score(Y_test, aux, average = 'macro'), recall_score(Y_test, aux, average = 'macro')

