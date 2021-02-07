from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from libs.activeLearningLib import activeLearningLib
from sklearn.ensemble import RandomForestClassifier
from opfython.models import SemiSupervisedOPF
from opfython.models import SupervisedOPF
from modAL.models import ActiveLearner
import opfython.math.general as g
import numpy as np
import time

class pipeline:

    activeLearningLibObject = activeLearningLib()

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

    # function to select samples
    def selectSamples(self, method, X, Y, K, first, learner):
        correctedLabels = 0
        # To measure the time
        t = time.time()
        # If to get the Active Learning technique selected
        if method == "Random":
            if first == True:
                # This function select Random Data
                selected_X, selected_Y, X, Y = self.activeLearningLibObject.randomActiveLearning(K, X, Y)
                correctedLabels = K
            else:
                # This function select Random Data
                selected_X, selected_Y, X, Y = self.activeLearningLibObject.randomActiveLearning(K, X, Y)
                for i in range(0, K):
                    # compare predicted label with true label to count correcteds
                    if np.array_equal(learner.predict([selected_X[i]]), selected_Y[i]) == False:
                            correctedLabels += 1
        if method in ["Entropy Sampling", "Margin Sampling", "Uncertainty Sampling", "Average Confidence", "RDS", "MST-BE"]:
            if first == True:
                if method in ["RDS", "MST-BE"]:
                    idx, root_idx, selected_X, selected_Y, X, Y = activeLearningLib_Object.get_samples(
                        X,
                        Y,
                        n_clusters = K,
                        strategy = method
                    )
                else:
                    # This function select the firsts roots
                    selected_X, selected_Y, X, Y = self.activeLearningLibObject.getRootSamples(K, X, Y)
                    correctedLabels = K
            else:
                for i in range(0, K):
                    query_idx, query_sample = learner.query(X)
                    # compare predicted label with true label to count correcteds
                    if learner.predict(X[query_idx]) != Y[query_idx]:
                        correctedLabels += 1
                    if i == 0:
                        selected_X = X[query_idx].flatten()
                        selected_Y = Y[query_idx].flatten()
                        X = np.delete(X, query_idx, axis = 0)
                        Y = np.delete(Y, query_idx, axis = 0)
                    else:
                        selected_X = np.vstack((selected_X, X[query_idx]))
                        selected_Y = np.vstack((selected_Y, Y[query_idx]))
                        X = np.delete(X, query_idx, axis = 0)
                        Y = np.delete(Y, query_idx, axis = 0)
        return round((time.time() - t), 3), selected_X, selected_Y, X, Y, correctedLabels

    def createLearner(self, method, x, y, learner, first):
        if method in ["Entropy Sampling", "Margin Sampling", "Uncertainty Sampling", "Average Confidence", "RDS", "MST-BE"]:
                    if first:
                        if method == "Entropy Sampling":
                            learner = ActiveLearner(
                                estimator = RandomForestClassifier(),
                                query_strategy = entropy_sampling,
                                X_training = x, y_training = y.astype(int)
                            )
                        if method == "Margin Sampling":
                            learner = ActiveLearner(
                                estimator = RandomForestClassifier(),
                                query_strategy = margin_sampling,
                                X_training = x, y_training = y.astype(int)
                            )
                        if method == "Uncertainty Sampling":
                            learner = ActiveLearner(
                                estimator = RandomForestClassifier(),
                                query_strategy = uncertainty_sampling,
                                X_training = x, y_training = y.astype(int)
                            )
                        if method == "Average Confidence":
                            learner = ActiveLearner(
                                estimator = RandomForestClassifier(),
                                query_strategy = avg_confidence,
                                X_training = x, y_training = y.astype(int)
                            )
                        if method == "RDS":
                            learner = ActiveLearner(
                                estimator = RandomForestClassifier(),
                                query_strategy = root_distance_based_selection_strategy,
                                X_training = x, y_training = y.astype(int)
                            )
                        if method == "MST-BE":
                            learner = ActiveLearner(
                                estimator = RandomForestClassifier(),
                                query_strategy = disagree_labels_edges_idx_query_strategy,
                                X_training = x, y_training = y.astype(int)
                            )
                    else:
                        learner.teach(x, y.astype(int))
        else:
            learner = "none"
        return learner
    
    def calcWrongPercentage(self, ss_predict, unlabeled_Y):
        count = 0
        for predict, unlabeled in zip(ss_predict, unlabeled_Y):
            if predict != unlabeled:
                count += 1
        return round((count * 100) / len(ss_predict), 2)

    def run(self, technique, easy_X, easy_Y, hard_X, hard_Y, test_X, test_Y, k_hardSamples, k_easySamples, iteration):
        # ---------- PIPELINE ----------
        print("")
        print("Iniciando Pipeline..")
        print("")

        # final variables
        ssmodel_accuracy = []
        ssmodel_corrected = []
        fullmodel_accuracy = []
        fullmodel_corrected = []
        hard_time_to_select = []
        easy_time_to_select = []
        ssmodel_knowClass = []
        fullmodel_knowClass = []
        wrong_percentage = []
        easy_X_bkp = easy_X; easy_Y_bkp = easy_Y; hard_X_bkp = hard_X; hard_Y_bkp = hard_Y

        # For to iterate over scenarios
        for (method_hard, method_easy) in zip([technique] * 2, ["Random", technique]):
            
            # for variables
            ss_model_score = []
            ss_model_corrected = []
            full_model_score = []
            full_model_corrected = []
            hardTimeToSelect = []
            easyTimeToSelect = []
            ss_know_class = []
            full_know_class = []
            wrongPercentages = []

            # Recover complete dataset
            easy_X = easy_X_bkp; easy_Y = easy_Y_bkp; hard_X = hard_X_bkp; hard_Y = hard_Y_bkp
            print("Métodos de Aprendizado Ativo: Dataset Hard {} / Dataset Easy {}".format(method_hard, method_easy))
            print("")

            # For to control iterations number
            for i in range(0, iteration):
                print("===== Iteração {} =====".format(i + 1))
                print("")

                # Selecting samples with Active Learning from Hard Dataset
                timeToSelect_hard, selected_hard_X, selected_hard_Y, hard_X, hard_Y, hard_correctedLabels = self.selectSamples(
                    method_hard,
                    hard_X,
                    hard_Y,
                    k_hardSamples * 2 if i == 0 else k_hardSamples,
                    True if i == 0 else False,
                    ("none" if i == 0 else learner_hard) if method_hard != "Random" else ("none" if i == 0 else ssmodel),
                )

                # Append True Labeled Data to the Labeled Data
                if i == 0:
                    labeled_X = selected_hard_X
                    labeled_Y = selected_hard_Y
                else:
                    labeled_X = np.vstack((labeled_X, selected_hard_X))
                    labeled_Y = np.vstack((labeled_Y, selected_hard_Y))
                print("Samples Labeled: {} - Time to Select: {} - Corrected: {}".format(len(labeled_Y), timeToSelect_hard, hard_correctedLabels))

                # Learner Object to apply into hard pool
                learner_hard = self.createLearner(method_hard, labeled_X, labeled_Y, "none" if i == 0 else learner_hard, True if i == 0 else False)
                learner_easy = self.createLearner(method_easy, labeled_X, labeled_Y, "none" if i == 0 else learner_easy, True if i == 0 else False)

                # selecting samples with Active Learning from Easy Dataset
                timeToSelect_easy, selected_easy_X, selected_easy_Y, easy_X, easy_Y, easy_correctedLabels = self.selectSamples(
                    method_easy,
                    easy_X,
                    easy_Y,
                    k_easySamples * 2 if i == 0 else k_easySamples,
                    (True if i == 0 else False) if method_easy == "Random" else False,
                    ((ssmodel if i != 0 else "none") if method_easy == "Random" else learner_easy)
                )

                # append True Labeled Data to the Unlabeled Data
                if i == 0:
                    unlabeled_X = selected_easy_X
                    unlabeled_Y = selected_easy_Y
                else:
                    unlabeled_X = np.vstack((unlabeled_X, selected_easy_X))
                    unlabeled_Y = np.vstack((unlabeled_Y, selected_easy_Y))
                print("Samples Unlabeled: {} - Time to Select: {} - Corrected: {}".format(len(unlabeled_Y), timeToSelect_easy, easy_correctedLabels))

                # semi supervised classification with OPF Semi Supervised
                t = time.time()
                ssmodel = SemiSupervisedOPF(distance = 'log_squared_euclidean', pre_computed_distance = None)
                ssmodel.fit(labeled_X, labeled_Y.flatten().astype("int"), unlabeled_X)
                print("Semi Supervised Score: {}% - Time: {}".format(round(g.opf_accuracy(test_Y.flatten().astype("int"), ssmodel.predict(test_X)) * 100, 2), round((time.time() - t), 3)))
                ss_model_score.append(round(g.opf_accuracy(test_Y.flatten().astype("int"), ssmodel.predict(test_X)) * 100, 2))

                # join labeled data with unlabeled
                Z_dataset_X = np.vstack((labeled_X, unlabeled_X))
                Z_dataset_Y = np.hstack((labeled_Y.flatten(), unlabeled_Y.flatten()))

                # full supervised classification
                fullmodel = SupervisedOPF(distance = 'log_squared_euclidean', pre_computed_distance = None)
                fullmodel.fit(Z_dataset_X, Z_dataset_Y.flatten().astype("int"))
                print("Full Supervised Score: {}% - Time: {}".format(round(g.opf_accuracy(test_Y.flatten().astype("int"), fullmodel.predict(test_X)) * 100, 2), round((time.time() - t), 3)))
                full_model_score.append(round(g.opf_accuracy(test_Y.flatten().astype("int"), fullmodel.predict(test_X)) * 100, 2))

                # Predict Semi-Supervised Labels to See how many errors are propagating
                ss_predict = ssmodel.predict(unlabeled_X)
                wrongPercentage = self.calcWrongPercentage(ss_predict, unlabeled_Y)

                # List of corrected Labels by methods
                ss_model_corrected.append(hard_correctedLabels)
                full_model_corrected.append(hard_correctedLabels + easy_correctedLabels)
                # List of time's to select
                hardTimeToSelect.append(timeToSelect_hard)
                easyTimeToSelect.append(timeToSelect_easy)
                # List of known class
                ss_know_class.append(len(np.unique(labeled_Y)))
                full_know_class.append(len(np.unique(Z_dataset_Y)))
                # List of wrong percentages
                wrongPercentages.append(wrongPercentage)
                print("")

            # Append Results
            ssmodel_accuracy.append(ss_model_score)
            ssmodel_corrected.append(ss_model_corrected)
            fullmodel_accuracy.append(full_model_score)
            fullmodel_corrected.append(full_model_corrected)
            hard_time_to_select.append(hardTimeToSelect)
            easy_time_to_select.append(easyTimeToSelect)
            ssmodel_knowClass.append(ss_know_class)
            fullmodel_knowClass.append(full_know_class)
            wrong_percentage.append(wrongPercentages)

            print("==="*25)
            print("")
        
        return ssmodel_accuracy, ssmodel_corrected, fullmodel_accuracy, fullmodel_corrected, \
               hard_time_to_select, easy_time_to_select, ssmodel_knowClass, fullmodel_knowClass, wrong_percentage