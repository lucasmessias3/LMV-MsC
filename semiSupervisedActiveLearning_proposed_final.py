from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from libs.pcc import ParticleCompetitionAndCooperation
from libs.activeLearningLib import activeLearningLib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from opfython.models import SemiSupervisedOPF
from modAL.multilabel import avg_confidence
from modAL.models import ActiveLearner
from libs.utils import utils
import collections
import pandas as pd
import numpy as np
import warnings
import json
import sys
import time

warnings.filterwarnings("ignore")

# ========== HOW TO USE ==========

# Possibles datasets: larvae, eggs, protozoan, all-15, all-16
# Possibles architectures: 0 - Inception_resnet_v2, 1 - Inception_v3, 2 - Celso
# Possibles activeSelection: 0 - RS, 1 - KN
# Example: python mainFramework-V6.py larvae 2 0

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
        edges_left = activeLearningLibObject.unique_without_sorting(np.delete(idx, disagree_edges_idx)[::-1])
        for e in edges_left:
            if e not in labeled_idx and e not in query_idx:
                query_idx = np.append(query_idx, e)
            if query_idx.size >= n_instances:
                return query_idx[:n_instances], np.delete(idx, disagree_edges_idx)

    return query_idx, np.empty(0, int)

def createLearner(X_initial, Y_initial, method):

    if method == "Entropy Sampling":
        learner = ActiveLearner(
            estimator = MLPClassifier(),
            query_strategy = entropy_sampling,
            X_training = X_initial, y_training = Y_initial
        )
    elif method == "Margin Sampling":
        learner = ActiveLearner(
            estimator = MLPClassifier(),
            query_strategy = margin_sampling,
            X_training = X_initial, y_training = Y_initial
        )
    elif method == "Uncertainty Sampling":
        learner = ActiveLearner(
            estimator = MLPClassifier(),
            query_strategy = uncertainty_sampling,
            X_training = X_initial, y_training = Y_initial
        )
    elif method == "Average Confidence":
        learner = ActiveLearner(
            estimator = MLPClassifier(),
            query_strategy = avg_confidence,
            X_training = X_initial, y_training = Y_initial
        )
    elif method == "RDS":
        learner = ActiveLearner(
            estimator = MLPClassifier(),
            query_strategy = root_distance_based_selection_strategy,
            X_training = X_initial, y_training = Y_initial
        )
    elif method == "MST-BE":
        learner = ActiveLearner(
            estimator = MLPClassifier(),
            query_strategy = disagree_labels_edges_idx_query_strategy,
            X_training = X_initial, y_training = Y_initial
        )

    return learner

def getSamples(X, Y, K, method):
    if method in ["RDS", "MST-BE"]:
        idx, root_idx, X_initial, Y_initial, X_pool, Y_pool = activeLearningLibObject.get_samples(
            X,
            Y,
            n_clusters = K,
            strategy = method
        )
    else:
        if method != "rand":
            idx, X_initial, Y_initial, X_pool, Y_pool = activeLearningLibObject.getRootSamples(K, X, Y)
        else:
            idx, X_initial, Y_initial, X_pool, Y_pool = activeLearningLibObject.randomActiveLearning(K, X, Y)
    
    return idx, X_initial, Y_initial, X_pool, Y_pool

# ---------- MAIN ----------

# loading metadatafrom modAL.models import ActiveLearner
with open('/home/lucasmessias/MsC-Project/framework/tools/metadata.json') as json_file:
    metadata = json.load(json_file)

# loading libs
utilsObject = utils()
activeLearningLibObject = activeLearningLib()

# reading files
easy_X, easy_Y, hard_X, hard_Y, test_X, test_Y = utilsObject.readCSV(
    sys.argv[1],
    metadata[sys.argv[1]]["architecture"][int(sys.argv[2])],
    metadata[sys.argv[1]]["activeSelection"][int(sys.argv[3])]
)

# constraints
activeLearning = ["Entropy Sampling", "Margin Sampling", "Uncertainty Sampling", "RDS", "MST-BE"]
semiLearning = ["OPFSemi", "PCC"]
k_hardSamples = int(metadata[sys.argv[1]]["K"]/2)
k_easySamples = metadata[sys.argv[1]]["K"] - k_hardSamples
iteration = 101

# pipeline
print("\nComeçando Pipeline..\n")

accuracy_opf = []
accuracy_pcc = []
timeToTrain_opf = []
timeToTest_opf = []
timeToTrain_pcc = []
timeToTest_pcc = []
scenarioList = []
numSamples = []
knowClasses_opf = []
knowClasses_pcc = []
propagatedErrors_opf = []
propagatedErrors_pcc = []
correctedLabels_opf = []
correctedLabels_pcc = []
f1Score_opf = []
f1Score_pcc = []
precision_opf = []
precision_pcc = []
recall_opf = []
recall_pcc = []

for technique in activeLearning:
    for scenario in [[technique, "rand"], [technique, technique]]:
        print("\nHard Samples: {}, Easy Samples: {}\n".format(scenario[0], scenario[1]))

        idx_hard, X_initial_hard, Y_initial_hard, X_pool_hard, Y_pool_hard = getSamples(hard_X, hard_Y, k_hardSamples, scenario[0])
        idx_easy, X_initial_easy, Y_initial_easy, X_pool_easy, Y_pool_easy = getSamples(easy_X, easy_Y, k_easySamples, scenario[1])

        learner_hard = createLearner(X_initial_hard, Y_initial_hard, scenario[0])
        # if scenario[1] != "rand":
            # learner_easy = createLearner(X_initial_easy, Y_initial_easy, scenario[1])

        labeled_idx_hard = np.empty(0, int)
        labeled_idx_easy = np.empty(0, int)

        X_hardData = X_initial_hard
        Y_hardData = Y_initial_hard
        X_easyData = X_initial_easy
        Y_easyData = Y_initial_easy

        # OPF SEMI
        # opfModel = SemiSupervisedOPF(distance = 'log_squared_euclidean', pre_computed_distance = None)

        # PCC SEMI
        pccModel = ParticleCompetitionAndCooperation(n_neighbors = 32, pgrd = 0.6, delta_v = 0.35, max_iter = 1000)

        for run in np.arange(iteration):

            scenarioList.append(scenario[0] + " - " + scenario[1])
            numSamples.append(len(Y_hardData)*2)

            uniqueLabels = np.unique(Y_hardData)
            auxLabels = []
            for i in Y_hardData:
                auxLabels.append(np.where(uniqueLabels == i)[0])
            auxLabels = [x for xs in auxLabels for x in xs]

            # t = time.time()
            # opfModel.fit(X_hardData, Y_hardData.astype("int"), X_easyData)
            # timeToTrain_opf.append(time.time() - t)
            timeToTrain_opf.append(0)

            # propagatedOPF = np.asarray(opfModel.predict(X_easyData))

            # t = time.time()
            # opfPreds = np.asarray(opfModel.predict(test_X))
            # timeToTest_opf.append(time.time() - t)
            timeToTest_opf.append(0)

            t = time.time()
            pccModel.fit(np.vstack([X_hardData, X_easyData]), np.hstack([auxLabels, np.full(len(Y_easyData), -1)]))
            timeToTrain_pcc.append(time.time() - t)

            t = time.time()
            preds = np.array(pccModel.predict(np.hstack([auxLabels, np.full(len(Y_easyData), -1)])))
            pccPreds_preview = preds[np.hstack([auxLabels, np.full(len(Y_easyData), -1)]) == -1]
            auxLabels2 = []
            for i in pccPreds_preview:
                    auxLabels2.append(uniqueLabels[int(i)])
            model = MLPClassifier().fit(np.vstack([X_hardData, X_easyData]), np.hstack([Y_hardData.flatten(), auxLabels2]))
            pccPreds = model.predict(test_X)
            timeToTest_pcc.append(time.time() - t)

            propagatedErrors = 0
            for (i, j) in zip(Y_easyData, auxLabels2):
                if i != j:
                    propagatedErrors += 1
            propagatedErrors = (propagatedErrors * 100) / len(Y_easyData)

            # print("opf acc: ", accuracy_score(test_Y, opfPreds))
            print("pcc acc: ", accuracy_score(test_Y, pccPreds))
            print("know classes {} : {}".format(len(uniqueLabels), uniqueLabels))
            # print("opf propagated errors: ", np.sum(Y_easyData.flatten() != propagatedOPF) * 100 / len(Y_easyData))
            print("pcc propagated errors: ", propagatedErrors)
            print("")

            # accuracy_opf.append(accuracy_score(test_Y, opfPreds))
            # f1Score_opf.append(f1_score(test_Y, opfPreds, average = 'macro'))
            # precision_opf.append(precision_score(test_Y, opfPreds, average = 'macro'))
            # recall_opf.append(recall_score(test_Y, opfPreds, average = 'macro'))
            accuracy_opf.append(0)
            f1Score_opf.append(0)
            precision_opf.append(0)
            recall_opf.append(0)
            knowClasses_opf.append(len(uniqueLabels))


            accuracy_pcc.append(accuracy_score(test_Y, pccPreds))
            knowClasses_pcc.append(len(uniqueLabels))
            f1Score_pcc.append(f1_score(test_Y, pccPreds, average = 'macro'))
            precision_pcc.append(precision_score(test_Y, pccPreds, average = 'macro'))
            recall_pcc.append(recall_score(test_Y, pccPreds, average = 'macro'))

            # propagatedErrors_opf.append(np.sum(Y_easyData.flatten() != propagatedOPF) * 100 / len(Y_easyData))
            propagatedErrors_opf.append(0)
            propagatedErrors_pcc.append(propagatedErrors)

            kwargs_hard = dict()
            if k_hardSamples > len(idx_hard): break
            kwargs_hard = dict(idx = idx_hard, labeled_idx = labeled_idx_hard, y_root = Y_initial_hard)
            
            if scenario[0] in ["RDS", "MST-BE"]:
                query_idx_hard, idx_hard = learner_hard.query(X_pool_hard, n_instances = k_hardSamples, **kwargs_hard)

                counter = 0
                toSpecialistLabels = learner_hard.predict(X_pool_hard[query_idx_hard])
                for (x, y) in zip(toSpecialistLabels, Y_pool_hard[query_idx_hard]):
                    if x != y:
                        counter += 1
                correctedLabels_pcc.append(counter)
                correctedLabels_opf.append(0)

                X_hardData = np.vstack((X_hardData, X_pool_hard[query_idx_hard]))
                Y_hardData = np.vstack((Y_hardData, Y_pool_hard[query_idx_hard]))
                learner_hard.teach(X = X_pool_hard[query_idx_hard], y = Y_pool_hard[query_idx_hard])
            else:
                query_idx_hard, idx_hard = learner_hard.query(X_pool_hard, n_instances = k_hardSamples)
                
                counter = 0
                toSpecialistLabels = learner_hard.predict(X_pool_hard[query_idx_hard])
                for (x, y) in zip(toSpecialistLabels, Y_pool_hard[query_idx_hard]):
                    if x != y:
                        counter += 1
                correctedLabels_pcc.append(counter)
                correctedLabels_opf.append(0)

                X_hardData = np.vstack((X_hardData, X_pool_hard[query_idx_hard]))
                Y_hardData = np.vstack((Y_hardData, Y_pool_hard[query_idx_hard]))
                learner_hard.teach(X = X_pool_hard[query_idx_hard], y = Y_pool_hard[query_idx_hard])
                X_pool_hard, Y_pool_hard = np.delete(X_pool_hard, query_idx_hard, axis = 0), np.delete(Y_pool_hard, query_idx_hard, axis = 0)

            if query_idx_hard is None or len(query_idx_hard) < k_hardSamples: break
            labeled_idx_hard = np.append(labeled_idx_hard, query_idx_hard)

            print("pcc labeled samples by specialist: ", counter)

            if scenario[1] != "rand":
                kwargs_easy = dict()
                kwargs_easy = dict(idx = idx_easy, labeled_idx = labeled_idx_easy, y_root = Y_initial_easy)
                
                if scenario[1] in ["RDS", "MST-BE"]:
                    query_idx_easy, idx_easy = learner_hard.query(X_pool_easy, n_instances = k_easySamples, **kwargs_easy)
                    # query_idx_easy, idx_easy = learner_easy.query(X_pool_easy, n_instances = k_easySamples, **kwargs_easy)
                    X_easyData = np.vstack((X_easyData, X_pool_easy[query_idx_easy]))
                    Y_easyData = np.vstack((Y_easyData, Y_pool_easy[query_idx_easy]))
                    # learner_easy.teach(X = X_pool_easy[query_idx_easy], y = Y_pool_easy[query_idx_easy])
                else:
                    query_idx_easy, idx_easy = learner_hard.query(X_pool_easy, n_instances = k_easySamples)
                    # query_idx_easy, idx_easy = learner_easy.query(X_pool_easy, n_instances = k_easySamples)
                    X_easyData = np.vstack((X_easyData, X_pool_easy[query_idx_easy]))
                    Y_easyData = np.vstack((Y_easyData, Y_pool_easy[query_idx_easy]))
                    # learner_easy.teach(X = X_pool_easy[query_idx_easy], y = Y_pool_easy[query_idx_easy])
                    X_pool_easy, Y_pool_easy = np.delete(X_pool_easy, query_idx_easy, axis = 0), np.delete(Y_pool_easy, query_idx_easy, axis = 0)
                if query_idx_easy is None or len(query_idx_easy) < k_easySamples: break
                labeled_idx_easy = np.append(labeled_idx_easy, query_idx_easy)
            else:
                query_idx_easy = np.random.choice(X_pool_easy.shape[0], k_easySamples, replace = False)
                X_easyData = np.vstack((X_easyData, X_pool_easy[query_idx_easy]))
                Y_easyData = np.vstack((Y_easyData, Y_pool_easy[query_idx_easy]))
                X_pool_easy, Y_pool_easy = np.delete(X_pool_easy, query_idx_easy, axis = 0), np.delete(Y_pool_easy, query_idx_easy, axis = 0)
        
results = pd.DataFrame(
    list(
        zip(
            scenarioList, numSamples,
            accuracy_opf, timeToTrain_opf, timeToTest_opf, propagatedErrors_opf, knowClasses_opf, correctedLabels_opf, f1Score_opf, precision_opf, recall_opf,
            accuracy_pcc, timeToTrain_pcc, timeToTest_pcc, propagatedErrors_pcc, knowClasses_pcc, correctedLabels_pcc, f1Score_pcc, precision_pcc, recall_pcc
        )
    )
    , columns = [
        "scenario", "numSamples",
        "acc-opf", "timeTrain-opf", "timeTest-opf", "propagatedErrors_opf", "knowClasses_opf", "labeledSamples_opf", "f1Score_opf", "precision_opf", "recall_opf",
        "acc-pcc", "timeTrain-pcc", "timeTest-pcc", "propagatedErrors_pcc", "knowClasses_pcc", "labeledSamples_pcc", "f1Score_pcc", "precision_pcc", "recall_pcc"
        ]
)

results.to_csv("/home/lucasmessias/MsC-Project/framework/results/" + sys.argv[1] + "/pipeline2/" + metadata[sys.argv[1]]["architecture"][int(sys.argv[2])] + "-" + metadata[sys.argv[1]]["activeSelection"][int(sys.argv[3])] + ".csv", index = False)