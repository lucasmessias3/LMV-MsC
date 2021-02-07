from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from modAL.multilabel import avg_confidence
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from opfython.models import SupervisedOPF
from sklearn.svm import SVC
from modAL.models import ActiveLearner
from libs.activeLearningLib import activeLearningLib
from opfython.models import SupervisedOPF
import pandas as pd
import numpy as np
import warnings
import random
import time
import json
import os
warnings.filterwarnings("ignore")

# Used Methods:

# ModAL - https://modal-python.readthedocs.io/en/latest/content/query_strategies/uncertainty_sampling.html
# RDS
# MST-BE

activeLearningLib_Object = activeLearningLib()

def root_distance_based_selection_strategy(classifier, x, n_instances = 1, **kwargs):
    y_root = kwargs.get("y_root")
    dic = kwargs.get("idx")
    query_idx = np.empty(0, int)

    n_samples_left = sum(map(len, dic.values()))
    while (query_idx.size < n_samples_left):
        for l in dic:
            idx = dic[l]
            if idx.size == 0: continue
            pred = classifier.predict(x[idx])
            sel = idx[np.where(pred != y_root[l])]
            if sel.size != 0:
                query_idx = np.append(query_idx, sel[0])
                dic[l] = idx[np.where(idx != sel[0])]
            else:
                query_idx = np.append(query_idx, idx[-1])
                dic[l] = idx[:-1]
            if query_idx.size == n_instances or query_idx.size == n_samples_left:
                return query_idx, dic

    return query_idx, dic

def disagree_labels_edges_idx_query_strategy(classifier, x, n_instances = 1, step = 2, **kwargs):
    labeled_idx = kwargs.get("labeled_idx")
    idx = kwargs.get("idx")
    query_idx = np.empty(0, int)
    disagree_edges_idx = np.empty(0, int)
    r = int(len(idx) / step)
    for i in range(r):
        begin = i * step
        end = begin + step
        edge_idx = idx[begin:end]
        pred = classifier.predict(x[edge_idx])
        if np.all(np.in1d(pred[1:], pred[0], invert=True)):
            disagree_edges_idx = np.append(disagree_edges_idx, np.arange(begin, end))
            query_idx = np.append(query_idx,
                                  np.array([e for e in edge_idx if e not in labeled_idx and e not in query_idx]).astype(
                                      int))
        if query_idx.size >= n_instances:
            return query_idx[:n_instances], np.delete(idx, disagree_edges_idx)

    if query_idx.size < n_instances:
        edges_left = activeLearningLib_Object.unique_without_sorting(np.delete(idx, disagree_edges_idx)[::-1])
        for e in edges_left:
            if e not in labeled_idx and e not in query_idx:
                query_idx = np.append(query_idx, e)
            if query_idx.size >= n_instances:
                return query_idx[:n_instances], np.delete(idx, disagree_edges_idx)

    return query_idx, np.empty(0, int)

def activeLearning(method, X_train, Y_train, X_test, Y_test, K):

  interations = 101
  random.seed(0)
  
  # Define initial labels indexs to train classifier
  if method in ["RDS", "MST-BE"]:
    idx, root_idx, X_initial, Y_initial, X_pool, Y_pool = activeLearningLib_Object.get_samples(
      X_train,
      Y_train,
      n_clusters = int(len(np.unique(Y_train)) * 2),
      strategy = method
    )
    labeled_idx = np.empty(0, int)
  else:
    idx = np.asarray(random.sample(range(0, len(X_train)), k = K))
    X_initial, Y_initial = X_train[idx], Y_train[idx]
    X_pool, Y_pool = np.delete(X_train, idx, axis = 0), np.delete(Y_train, idx, axis = 0)

  # Initialize Active Learning Methods
  t = time.time()
  if method == "Entropy Sampling":
    learner = ActiveLearner(
        estimator = SVC(probability = True),
        query_strategy = entropy_sampling,
        X_training = X_initial, y_training = Y_initial
    )
  elif method == "Margin Sampling":
    learner = ActiveLearner(
        estimator = SVC(probability = True),
        query_strategy = margin_sampling,
        X_training = X_initial, y_training = Y_initial
    )
  elif method == "Uncertainty Sampling":
    learner = ActiveLearner(
        estimator = SVC(probability = True),
        query_strategy = uncertainty_sampling,
        X_training = X_initial, y_training = Y_initial
    )
  elif method == "Average Confidence":
    learner = ActiveLearner(
        estimator = SVC(probability = True),
        query_strategy = avg_confidence,
        X_training = X_initial, y_training = Y_initial
    )
  elif method == "RDS":
    learner = ActiveLearner(
        estimator = SVC(probability = True),
        # estimator = SupervisedOPF(distance = "log_squared_euclidean", pre_computed_distance = None),
        query_strategy = root_distance_based_selection_strategy,
        X_training = X_initial, y_training = Y_initial
    )
  elif method == "MST-BE":
    learner = ActiveLearner(
        estimator = SVC(probability = True),
        # estimator = SupervisedOPF(distance = "log_squared_euclidean", pre_computed_distance = None),
        query_strategy = disagree_labels_edges_idx_query_strategy,
        X_training = X_initial, y_training = Y_initial
    )
  timeToTrain = time.time() - t
  
  results = []

  labeledData_X = X_initial
  labeledData_Y = Y_initial
  
  for run in range(interations):

    if K > len(idx): break

    if method in ["RDS", "MST-BE"]:

      kwargs = dict()
      if K > len(idx): break
      kwargs = dict(idx = idx, labeled_idx = labeled_idx, y_root = Y_initial)

      t = time.time()
      query_idx, idx = learner.query(X_pool, n_instances = K, **kwargs)
      timeToSelect = time.time() - t

      if query_idx is None or len(query_idx) < K: break
      labeled_idx = np.append(labeled_idx, query_idx)

      predsCorrecteds = learner.predict(X_pool[query_idx])
      counter = 0
      for (x, y) in zip(predsCorrecteds, Y_pool[query_idx].flatten()):
        if x != y:
          counter += 1

      t = time.time()
      learner.teach(X = X_pool[query_idx], y = Y_pool[query_idx])
      timeToTrain = time.time() - t

      labeledData_X = np.vstack((labeledData_X, X_pool[query_idx]))
      labeledData_Y = np.vstack((labeledData_Y, Y_pool[query_idx]))
      t = time.time()
      # model = SupervisedOPF(distance = "log_squared_euclidean", pre_computed_distance = None)
      # trained_model = model.fit(labeledData_X, labeledData_Y.flatten().astype("int"))
      preds = learner.predict(X_test.values)
      timeToTest = time.time() - t

      acc = accuracy_score(Y_test, preds)
      f1score = f1_score(Y_test, preds, average = 'macro')
      precision = precision_score(Y_test, preds, average = 'macro')
      recall = recall_score(Y_test, preds, average = 'macro')
      knowClasses = len(set(preds.tolist()))

      print("Run {}: Acc: {}".format(run + 1, acc))
      print("Know Classes: {}".format(knowClasses))
      print("Corrected Labels: {}".format(counter))
      print("Time to Select: {}".format(timeToSelect))
    else:
      if run == 0:

        t = time.time()
        # model = SupervisedOPF(distance = "log_squared_euclidean", pre_computed_distance = None)
        # trained_model = model.fit(labeledData_X, labeledData_Y.flatten().astype("int"))
        preds = learner.predict(X_test.values)
        timeToTest = time.time() - t

        acc = accuracy_score(Y_test, preds)
        f1score = f1_score(Y_test, preds, average = 'macro')
        precision = precision_score(Y_test, preds, average = 'macro')
        recall = recall_score(Y_test, preds, average = 'macro')
        knowClasses = len(set(preds.tolist()))
        counter = len(Y_initial)
        timeToSelect = 0

        print("Run {}: Acc: {}".format(run + 1, acc))
        print("Know Classes: {}".format(knowClasses))
        print("Corrected Labels: {}".format(counter))
        print("Time to Select: {}".format(timeToSelect))
      else:
        try:
          t = time.time()
          query_idx, idx = learner.query(X_pool, n_instances = K)
          timeToSelect = time.time() - t
        except:
          timeToSelect = 0
          print("deu erro")
          break

        predsCorrecteds = learner.predict(X_pool[query_idx])
        counter = 0
        for (x, y) in zip(predsCorrecteds, Y_pool[query_idx].flatten()):
          if x != y:
            counter += 1

        t = time.time()
        learner.teach(X = X_pool[query_idx], y = Y_pool[query_idx])
        # X_pool, Y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(Y_pool, query_idx, axis=0)
        timeToTrain = time.time() - t

        # t = time.time()
        # preds = learner.predict(X_test)
        # timeToTest = time.time() - t

        labeledData_X = np.vstack((labeledData_X, X_pool[query_idx]))
        labeledData_Y = np.vstack((labeledData_Y, Y_pool[query_idx]))
        t = time.time()
        # model = SupervisedOPF(distance = "log_squared_euclidean", pre_computed_distance = None)
        # trained_model = model.fit(labeledData_X, labeledData_Y.flatten().astype("int"))
        preds = learner.predict(X_test.values)
        X_pool, Y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(Y_pool, query_idx, axis=0)
        timeToTest = time.time() - t

        acc = accuracy_score(Y_test, preds)
        f1score = f1_score(Y_test, preds, average = 'macro')
        precision = precision_score(Y_test, preds, average = 'macro')
        recall = recall_score(Y_test, preds, average = 'macro')
        knowClasses = len(set(preds.tolist()))

        print("Run {}: Acc: {}".format(run + 1, acc))
        print("Know Classes: {}".format(knowClasses))
        print("Corrected Labels: {}".format(counter))
        print("Time to Select: {}".format(timeToSelect))
    
    results.append([run + 1, K, np.round(timeToTrain, 2), np.round(timeToTest, 2), np.round(timeToSelect, 2), np.round(acc * 100, 2), np.round(f1score * 100, 2), np.round(precision * 100, 2), np.round(recall * 100, 2), knowClasses, counter])

  results_df = pd.DataFrame(
    results,
    columns = ["iteration", "k-value", "time-to-train", "time-to-test", "time-to-select", "accuracy", "f1-score", "precision", "recall", "knowClasses", "correctedLabels"]
  )

  return results_df

# loading metadata
with open('/home/lucasmessias/MsC-Project/framework/tools/metadata.json') as json_file:
  metadata = json.load(json_file)

for folder in os.listdir(metadata["dirPath"]):
  for file in os.listdir(metadata["dirPath"] + str(folder) + "/"):

    # Log file to save terminal output
    # sys.stdout = open(metadata["resultsPath"] + str(folder) + "/supervised/terminal.log","w")

    if folder in ["all-15", "all-16"]:
    # if folder in ["all-15"]:

      print("\nWorking on: {}".format(str(folder) + "/" + str(file)))

      # Load CSV
      df = pd.read_csv(metadata["dirPath"] + str(folder) + "/" + str(file), header = None)

      # Loop for 6 active learning methods
      for method in ["Entropy Sampling", "Margin Sampling", "Uncertainty Sampling", "RDS", "MST-BE"]:
      # for method in ["RDS", "MST-BE"]:

        print("\nUsing Method {}".format(method), end = "\n\n")

        results = pd.DataFrame(columns = ["iteration", "k-value", "time-to-train", "time-to-test", "time-to-select", "accuracy", "f1-score", "precision", "recall", "knowClasses", "correctedLabels"])

        # To do 10 runs with the same seeds
        for run in metadata["seeds"][:1]:
          X_train, X_test, Y_train, Y_test = train_test_split(
            df.iloc[:, :-1],
            df.iloc[:, -1:].astype("int"),
            test_size = 0.2,
            random_state = run,
            stratify = df.iloc[:, -1:].astype("int")
          )

          results = results.append(activeLearning(
            method = method,
            X_train = X_train.values,
            Y_train = Y_train.values,
            X_test = X_test,
            Y_test = Y_test,
            K = metadata[folder]["K"]
          ))

        # Save results into a CSV File
        results.to_csv(metadata["resultsPath"] + str(folder) + "/active/" + str(file)[:-4] + "_" + str(method) + "_results.csv", index = False)

      # Save Log and close file
      # sys.stdout.close()