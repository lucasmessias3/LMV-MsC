from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from libs.pcc import ParticleCompetitionAndCooperation
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from opfython.models import SemiSupervisedOPF
import pandas as pd
import numpy as np
import json
import time
import os
import warnings
warnings.filterwarnings("ignore")

# Used Methods:

# OPFSemi - https://github.com/gugarosa/opfython
# Particle Competition and Cooperation - https://github.com/caiocarneloz/pycc
# Label Spreading - https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html
# Label Propagation - https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html

def classify(run, train_size, classifier, X_sup, Y_sup, X_unsup, Y_unsup, X_test, Y_test):

  if classifier == "OPFSemi":
    model = SemiSupervisedOPF(distance = 'log_squared_euclidean', pre_computed_distance = None)

    t = time.time()
    model.fit(X_sup, Y_sup.flatten().astype("int"), X_unsup)
    timeToTrain = time.time() - t

    t = time.time()
    preds = np.asarray(model.predict(X_test))
    timeToTest = time.time() - t

  elif classifier == "PCC":
    data = np.vstack([X_sup, X_test])
    labels = np.hstack([Y_sup.flatten(), Y_test.flatten()])
    print("\n Unique Labels of PCC: ", np.unique(labels))
    masked_labels = np.hstack([Y_sup.flatten(), np.full(len(Y_test), -1)])

    model = ParticleCompetitionAndCooperation(n_neighbors = 32, pgrd = 0.6, delta_v = 0.35, max_iter = 1000)
    
    t = time.time()
    model.fit(data, masked_labels)
    timeToTrain = time.time() - t

    t = time.time()
    preds = np.array(model.predict(masked_labels))
    timeToTest = time.time() - t

    labels = np.array(labels[masked_labels == -1]).astype("int")
    preds = preds[masked_labels == -1]

  acc = accuracy_score(Y_test, preds)
  f1score = f1_score(Y_test, preds, average = 'macro')
  precision = precision_score(Y_test, preds, average = 'macro')
  recall = recall_score(Y_test, preds, average = 'macro')

  print("Train-Size: {} TimeTrain: {} TimeTest: {} Acc: {} F1Score: {} Precision: {} Recall {}".format(
    np.round(train_size, 1), np.round(timeToTrain, 2), np.round(timeToTest, 2), np.round(acc * 100, 2), np.round(f1score * 100, 2), np.round(precision * 100, 2), np.round(recall * 100, 2)
  ))

  return [np.round(train_size, 1), np.round(timeToTrain, 2), np.round(timeToTest, 2), np.round(acc * 100, 2), np.round(f1score * 100, 2), np.round(precision * 100, 2), np.round(recall * 100, 2)]


# loading metadata
with open('/home/lucasmessias/MsC-Project/framework/tools/metadata.json') as json_file:
  metadata = json.load(json_file)

for folder in os.listdir(metadata["dirPath"]):
  for file in os.listdir(metadata["dirPath"] + str(folder) + "/"):

    # Log file to save terminal output
    # sys.stdout = open(metadata["resultsPath"] + str(folder) + "/semi/terminal.log","w")

    # if folder in ["all-15", "all-16"]:
    if (folder in ["all-16"]) & (file == "inception_v3.csv"):

      print("\nWorking on: {}".format(str(folder) + "/" + str(file)))

      # Load CSV
      df = pd.read_csv(metadata["dirPath"] + str(folder) + "/" + str(file), header = None)

      # Loop for 2 classifiers
      # for classifier in ["OPFSemi", "PCC"]:
      for classifier in ["OPFSemi", "PCC"]:

        print("\nUsing Classifier {}".format(classifier))

        results = []

        # To do 10 runs with the same seeds
        for run in metadata["seeds"][:1]:

          X_train, X_test, Y_train, Y_test = train_test_split(
            df.iloc[:, :-1],
            df.iloc[:, -1:].astype("int"),
            test_size = 0.2,
            random_state = run,
            stratify = df.iloc[:, -1:].astype("int")
          )

          # To variate percent of propagated values
          for percent in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            X_sup, X_unsup, Y_sup, Y_unsup = train_test_split(
              X_train,
              Y_train,
              test_size = percent,
              random_state = run,
              stratify = Y_train
            )
        
            # Classify
            results.append(
              classify(
                run = run,
                train_size = 1 - percent, 
                classifier = classifier,
                X_sup = X_sup.values,
                Y_sup = Y_sup.values,
                X_unsup = X_unsup.values,
                Y_unsup = Y_unsup.values,
                X_test = X_test.values,
                Y_test = Y_test.values
              )
            )

        # Save results in a Pandas DataFrame (each row is a run)
        results_df = pd.DataFrame(results, columns = ["train-size", "time-to-train", "time-to-test", "accuracy", "f1-score", "precision", "recall"])

        # Print DataFrame and some statistics to Debug
        print("")
        print(results_df.head(100))
        print("\nacc mean ", results_df["accuracy"].mean())
        print("acc std ", results_df["accuracy"].std())

        # Save results into a CSV File
        results_df.to_csv(metadata["resultsPath"] + str(folder) + "/semi/" + str(file)[:-4] + "_" + str(classifier) + "_results.csv", index = False)

      # # Save Log and close file
      # # sys.stdout.close()