from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from opfython.models import SupervisedOPF
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import warnings
import time
import json
import os
import sys
warnings.filterwarnings("ignore")

# Used Methods:

# OPF - https://github.com/gugarosa/opfython
# SVM - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# Random Forest - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Neural Net - https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

def classify(classifier, X_train, Y_train, X_test, Y_test):

  if classifier == "SVM":
    model = SVC()
  elif classifier == "RF":
    model = RandomForestClassifier()
  elif classifier == "NNET":
    model = MLPClassifier()
  elif classifier == "OPF":
    model = SupervisedOPF(distance = "log_squared_euclidean", pre_computed_distance = None)

  if classifier in ["SVM", "RF", "NNET"]:
    t = time.time()
    trained_model = model.fit(X_train, Y_train)
    timeToTrain = time.time() - t

    t = time.time()
    preds = model.predict(X_test)
    timeToTest = time.time() - t
  else:
    t = time.time()
    trained_model = model.fit(X_train.values, Y_train.values.flatten().astype("int"))
    timeToTrain = time.time() - t

    t = time.time()
    preds = model.predict(X_test.values)
    timeToTest = time.time() - t

  acc = accuracy_score(Y_test, preds)
  f1score = f1_score(Y_test, preds, average = 'macro')
  precision = precision_score(Y_test, preds, average = 'macro')
  recall = recall_score(Y_test, preds, average = 'macro')

  return [np.round(timeToTrain, 2), np.round(timeToTest, 2), np.round(acc * 100, 2), np.round(f1score * 100, 2), np.round(precision * 100, 2), np.round(recall * 100, 2)]

# loading metadata
with open('/home/lucasmessias/MsC-Project/framework/tools/metadata.json') as json_file:
  metadata = json.load(json_file)

for folder in os.listdir(metadata["dirPath"]):
  for file in os.listdir(metadata["dirPath"] + str(folder) + "/"):

    # Log file to save terminal output
    # sys.stdout = open(metadata["resultsPath"] + str(folder) + "/supervised/terminal.log","w")

    # if folder in ["all-15", "all-16"]:
    if (folder in ["all-16"]) & (file == "inception_v3.csv"):

      print("\nWorking on: {}".format(str(folder) + "/" + str(file)))

      # Load CSV
      df = pd.read_csv(metadata["dirPath"] + str(folder) + "/" + str(file), header = None)

      # Loop for 4 classifiers
      # for classifier in ["SVM", "RF", "NNET", "OPF"]:
      for classifier in ["OPF"]:

        print("\nUsing Classifier {}".format(classifier), end = "\n\n")

        results = []

        # To do 10 runs with the same seeds
        for run in metadata["seeds"]:
          X_train, X_test, Y_train, Y_test = train_test_split(
            df.iloc[:, :-1],
            df.iloc[:, -1:].astype("int"),
            test_size = 0.2,
            random_state = run,
            stratify = df.iloc[:, -1:].astype("int")
          )
        
          # Classify
          results.append(
            classify(
              classifier = classifier,
              X_train = X_train,
              Y_train = Y_train,
              X_test = X_test,
              Y_test = Y_test
            )
          )

          print(results)

        # Save results in a Pandas DataFrame (each row is a run)
        results_df = pd.DataFrame(results, columns = ["time-to-train", "time-to-test", "accuracy", "f1-score", "precision", "recall"])

        # Print DataFrame and some statistics to Debug
        print(results_df.head(50))
        print("\nacc mean ", results_df["accuracy"].mean())
        print("acc std ", results_df["accuracy"].std())

        # Save results into a CSV File
        results_df.to_csv(metadata["resultsPath"] + str(folder) + "/supervised/" + str(file)[:-4] + "_" + str(classifier) + "_results.csv", index = False)

      # Save Log and close file
      # sys.stdout.close()