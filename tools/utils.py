from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tkinter import filedialog
import pandas as pd
import numpy as np
import warnings
import arff
import time

warnings.filterwarnings("ignore")

class utils:

    def countUniqueLabels(self, trainData, testData, type):
        if type == "normalSplit":
            print("Disposição das amostras no Conjunto de Treinamento:")
        else:
            print("Disposição das amostras no Conjunto Hard:")
        # Counts the unique samples of each class from the training data
        unique, counts = np.unique(trainData.values, return_counts = True)
        for i in zip(unique, counts):
            print("      Número de amostras da Classe {} : {}".format(i[0], i[1]))
        if type == "normalSplit":
            print("Disposição das amostras no Conjunto de Teste:")
        else:
            print("")
            print("Disposição das amostras no Conjunto Easy:")
        # Counts the unique samples of each class from the test data
        unique, counts = np.unique(testData.values, return_counts = True)
        for i in zip(unique, counts):
            print("      Número de amostras da Classe {} : {}".format(i[0], i[1]))

    # Function to read Data
    # def readCSV(self, dataset, architecture, activeTechnique):
    #     # To measure the time
    #     t = time.time()
    #     # Read csv into Pandas
    #     easy_X = pd.read_csv("/home/lucasmessias/MsC-Project/framework/data/" + dataset + "/" + architecture + "/" +
    #         activeTechnique + "/easy_X.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
    #     easy_Y = pd.read_csv("/home/lucasmessias/MsC-Project/framework/data/" + dataset + "/" + architecture + "/" +
    #         activeTechnique + "/easy_Y.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
    #     hard_X = pd.read_csv("/home/lucasmessias/MsC-Project/framework/data/" + dataset + "/" + architecture + "/" +
    #         activeTechnique + "/hard_X.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
    #     hard_Y = pd.read_csv("/home/lucasmessias/MsC-Project/framework/data/" + dataset + "/" + architecture + "/" +
    #         activeTechnique + "/hard_Y.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
    #     test_X = pd.read_csv("/home/lucasmessias/MsC-Project/framework/data/" + dataset + "/" + architecture + "/" +
    #         activeTechnique + "/test_X.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
    #     test_Y = pd.read_csv("/home/lucasmessias/MsC-Project/framework/data/" + dataset + "/" + architecture + "/" +
    #         activeTechnique + "/test_Y.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
    #     print('===' * 25)
    #     print("")
    #     print("===== Dataset escolhido: {}".format(dataset))
    #     print("===== Arquitetura: {}".format(architecture))
    #     print("===== Seleção Ativa: {}".format(activeTechnique))
    #     print("")
    #     print('Conjunto Hard: {} Amostras - Conjunto Easy: {} Amostras'.format(len(hard_Y), len(easy_Y)))
    #     print('')
    #     # Counts the unique samples of each class
    #     self.countUniqueLabels(easy_Y, hard_Y, "Splitted")
    #     print("")
    #     print("Tempo gasto para carregar todos conjuntos: {}".format(round((time.time() - t), 3)))
    #     print("")
    #     print('===' * 25)
    #     return easy_X.values, easy_Y.values, hard_X.values, hard_Y.values, test_X.values, test_Y.values

    # Function to read Data
    def readCSV(self, dataset, architecture, activeTechnique):
        # To measure the time
        t = time.time()
        # Read csv into Pandas
        easy_X = pd.read_csv("/home/lucasmessias/MsC-Project/framework/pipelineData/" + dataset + "/" + architecture + "/" +
            activeTechnique + "/easy_X.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
        easy_Y = pd.read_csv("/home/lucasmessias/MsC-Project/framework/pipelineData/" + dataset + "/" + architecture + "/" +
            activeTechnique + "/easy_Y.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
        hard_X = pd.read_csv("/home/lucasmessias/MsC-Project/framework/pipelineData/" + dataset + "/" + architecture + "/" +
            activeTechnique + "/hard_X.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
        hard_Y = pd.read_csv("/home/lucasmessias/MsC-Project/framework/pipelineData/" + dataset + "/" + architecture + "/" +
            activeTechnique + "/hard_Y.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
        test_X = pd.read_csv("/home/lucasmessias/MsC-Project/framework/pipelineData/" + dataset + "/" + architecture + "/" +
            activeTechnique + "/test_X.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
        test_Y = pd.read_csv("/home/lucasmessias/MsC-Project/framework/pipelineData/" + dataset + "/" + architecture + "/" +
            activeTechnique + "/test_Y.csv").rename(columns = {"Unnamed: 0" : "index"}).set_index("index")
        print('===' * 25)
        print("")
        print("===== Dataset escolhido: {}".format(dataset))
        print("===== Arquitetura: {}".format(architecture))
        print("===== Seleção Ativa: {}".format(activeTechnique))
        print("")
        print('Conjunto Hard: {} Amostras - Conjunto Easy: {} Amostras'.format(len(hard_Y), len(easy_Y)))
        print('')
        # Counts the unique samples of each class
        self.countUniqueLabels(hard_Y, easy_Y, "Splitted")
        print("")
        print("Tempo gasto para carregar todos conjuntos: {}".format(round((time.time() - t), 3)))
        print("")
        print('===' * 25)
        return easy_X.values, easy_Y.values, hard_X.values, hard_Y.values, test_X.values, test_Y.values

    def splitData(self, df):
        # X get the features and Y the labels
        X = df.iloc[:, 0:-1]
        Y = df.iloc[:, -1:].astype("int")
        # Code the labels, changing from categorical to integer
        # for use in classifiers IF NEEDED
        # try:
        #     le = preprocessing.LabelEncoder()
        #     le = le.fit(Y.values)
        #     Y = pd.Series(le.transform(Y.values))
        #     print("passou aqui")
        # except:
        #     pass
        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size = 0.2,
            random_state = 42,
            stratify = Y
        )
        return X_train, X_test, Y_train, Y_test

    def supervisedProcess(self, df, file, dataset, supervisedClassifiers):
        dfAux = pd.DataFrame()

        dfAux["dataset"] = [dataset]
        dfAux["architecture"] = ["inception-resnet-v2" if "resnet" in file else "inception-v3"]
        dfAux["classifier"] = [
            supervisedClassifiers[0] if supervisedClassifiers[0] in file else(
            supervisedClassifiers[1] if supervisedClassifiers[1] in file else(
                supervisedClassifiers[2] if supervisedClassifiers[2] in file else(supervisedClassifiers[3])))
        ]
        dfAux["avgTimeTrain"] = [df["time-to-train"].mean()]
        dfAux["stdTimeTrain"] = [df["time-to-train"].std()]
        dfAux["avgTimeTest"] = [df["time-to-test"].mean()]
        dfAux["stdTimeTest"] = [df["time-to-test"].std()]
        dfAux["avgAcc"] = [df["accuracy"].mean()]
        dfAux["stdAcc"] = [df["accuracy"].std()]
        dfAux["avgF1Score"] = [df["f1-score"].mean()]
        dfAux["stdF1Score"] = [df["f1-score"].std()]
        dfAux["avgPrecision"] = [df["precision"].mean()]
        dfAux["stdPrecision"] = [df["precision"].std()]
        dfAux["avgRecall"] = [df["recall"].mean()]
        dfAux["stdRecall"] = [df["recall"].std()]

        return dfAux

    def semiProcess(self, df, file, dataset, semiClassifiers):
        dfAux = pd.DataFrame()

        dfAux["dataset"] = [dataset]*len(df)
        dfAux["architecture"] = ["inception-resnet-v2" if "resnet" in file else "inception-v3"]*len(df)
        dfAux["classifier"] = [semiClassifiers[0] if semiClassifiers[0] in file else semiClassifiers[1]]*len(df)
        dfAux["train-size"] = df["train-size"]
        dfAux["avgTimeTrain"] = df["time-to-train"]
        dfAux["stdTimeTrain"] = df["time-to-train"]
        dfAux["avgTimeTest"] = df["time-to-test"]
        dfAux["stdTimeTest"] = df["time-to-test"]
        dfAux["avgAcc"] = df["accuracy"]
        dfAux["stdAcc"] = df["accuracy"]
        dfAux["avgF1Score"] = df["f1-score"]
        dfAux["stdF1Score"] = df["f1-score"]
        dfAux["avgPrecision"] = df["precision"]
        dfAux["stdPrecision"] = df["precision"]
        dfAux["avgRecall"] = df["recall"]
        dfAux["stdRecall"] = df["recall"]

        return dfAux

    def activeProcess(self, df, file, dataset, activeClassifiers):
        dfAux = pd.DataFrame()

        dfAux["dataset"] = [dataset]*len(df)
        dfAux["architecture"] = ["inception-resnet-v2" if "resnet" in file else "inception-v3"]*len(df)
        dfAux["classifier"] = [
            activeClassifiers[0] if activeClassifiers[0] in file else(
            activeClassifiers[1] if activeClassifiers[1] in file else(
            activeClassifiers[2] if activeClassifiers[2] in file else(
            activeClassifiers[3] if activeClassifiers[3] in file else(
            activeClassifiers[4] if activeClassifiers[4] in file else(activeClassifiers[5])))))]*len(df)
        dfAux["iteration"] = df["iteration"]
        dfAux["k-value"] = df["k-value"]
        dfAux["avgTimeTrain"] = df["time-to-train"]
        dfAux["stdTimeTrain"] = df["time-to-train"]
        dfAux["avgTimeTest"] = df["time-to-test"]
        dfAux["stdTimeTest"] = df["time-to-test"]
        dfAux["avgAcc"] = df["accuracy"]
        dfAux["stdAcc"] = df["accuracy"]
        dfAux["avgF1Score"] = df["f1-score"]
        dfAux["stdF1Score"] = df["f1-score"]
        dfAux["avgPrecision"] = df["precision"]
        dfAux["stdPrecision"] = df["precision"]
        dfAux["avgRecall"] = df["recall"]
        dfAux["stdRecall"] = df["recall"]

        return dfAux

    def plotSupervised(self, df, folder):

        fig, axs = plt.subplots(3, 2, figsize = (15, 10))
        index = np.arange(len(np.unique(df.classifier.values)))

        for (ax, ay, title) in zip(
        [0, 0, 1, 1, 2, 2],
        [0, 1, 0, 1, 0, 1],
        ["TimeTrain", "TimeTest", "Acc", "F1Score", "Precision", "Recall"]):
            for (architecture, color, width) in zip(
                ["inception-resnet-v2", "inception-v3"],
                ["#7293cb", "#e1974c"],
                [-0.10, 0.10]):
                axs[ax, ay].bar(
                    index + width,
                    df[df["architecture"] == architecture]["avg" + title].values,
                    width = 0.20,
                    yerr = df[df["architecture"] == architecture]["std" + title].values,
                    label = architecture,
                    color = color
                )
                for (x, y) in zip(index + width, df[df["architecture"] == architecture]["avg" + title].values):
                    label = "{:.1f}".format(y)
                    axs[ax, ay].annotate(
                        label, (x, y), textcoords = "offset points", xytext = (0, 3), ha = "center", weight = "bold", fontsize = 8
                )
            axs[ax, ay].set_title(title, fontsize = 16, fontweight = "bold")
            axs[ax, ay].set_xlabel("Classifier's", fontsize = 10)
            axs[ax, ay].legend(fontsize = "small", ncol = 2)
            axs[ax, ay].set_ylim([0, df["avg" + title].values.max() * 1.3])
            axs[ax, ay].set_xticks(index)
            axs[ax, ay].set_xticklabels(np.unique(df.classifier.values), rotation = 45, ha = "right")
        fig.tight_layout()
        fig.subplots_adjust(top = 0.95)
        fig.suptitle(
            "Dataset: " + str(folder),
            horizontalalignment = "left",
            x = 0.03,
            fontsize = 15,
            weight = "bold"
        )

        plt.savefig("/home/lucasmessias/MsC-Project/framework/results/" + folder + "/supervised-results.png")

    def plotSemi(self, df2, folder):

        trainSize = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for size in trainSize:

            fig, axs = plt.subplots(3, 2, figsize = (15, 10))
            index = np.arange(len(np.unique(df2.classifier.values)))

            for (ax, ay, title) in zip(
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 1, 0, 1],
            ["TimeTrain", "TimeTest", "Acc", "F1Score", "Precision", "Recall"]):
                for (architecture, color, width) in zip(
                    ["inception-resnet-v2", "inception-v3"],
                    ["#7293cb", "#e1974c"],
                    [-0.10, 0.10]):
                    axs[ax, ay].bar(
                        index + width,
                        df2[(df2["architecture"] == architecture) & (df2["train-size"] == size)]["avg" + title].values,
                        width = 0.20,
                        # yerr = df2[(df2["architecture"] == architecture) & (df2["train-size"] == size)]["std" + title].values,
                        label = architecture,
                        color = color
                    )
                    for (x, y) in zip(index + width, df2[(df2["architecture"] == architecture) & (df2["train-size"] == size)]["avg" + title].values):
                        label = "{:.1f}".format(y)
                        axs[ax, ay].annotate(
                            label, (x, y), textcoords = "offset points", xytext = (0, 3), ha = "center", weight = "bold", fontsize = 8
                    )
                axs[ax, ay].set_title(title, fontsize = 16, fontweight = "bold")
                axs[ax, ay].set_xlabel("Classifier's", fontsize = 10)
                axs[ax, ay].legend(fontsize = "small", ncol = 2)
                axs[ax, ay].set_ylim([0, df2[df2["train-size"] == size]["avg" + title].values.max() * 1.3])
                axs[ax, ay].set_xticks(index)
                axs[ax, ay].set_xticklabels(np.unique(df2.classifier.values), rotation = 45, ha = "right")
            fig.tight_layout()
            fig.subplots_adjust(top = 0.92)
            fig.suptitle(
                "Dataset: " + str(folder) + "\nTrain-Size: " + str(size),
                horizontalalignment = "left",
                x = 0.03,
                fontsize = 15,
                weight = "bold"
            )

            plt.savefig("/home/lucasmessias/MsC-Project/framework/results/" + str(folder) + "/semi-results-" + str(size).replace(".", "_") + ".png")
    def plotActive(self, df3, folder):
    
        for architecture in ["inception-resnet-v2", "inception-v3"]:

            fig, axs = plt.subplots(3, 2, figsize = (15, 10))
            index = np.arange(len(np.unique(df3.iteration.values)))

            for (ax, ay, title) in zip(
                [0, 0, 1, 1, 2, 2],
                [0, 1, 0, 1, 0, 1],
                ["TimeTrain", "TimeTest", "Acc", "F1Score", "Precision", "Recall"]):
                for (classifier, color) in zip(
                ["Entropy Sampling", "Margin Sampling", "Uncertainty Sampling", "Average Confidence", "RDS", "MST-BE"],
                ["red", "blue", "green", "yellow", "orange", "grey"]
                ):
                    axs[ax, ay].plot(
                        index,
                        df3[(df3["architecture"] == architecture) & (df3["classifier"] == classifier)]["avg" + title].values,
                        linewidth = 3,
                        label = classifier,
                        color = color,
                        alpha = 0.8
                    )
                axs[ax, ay].set_title(title, fontsize = 16, fontweight = "bold")
                axs[ax, ay].set_xlabel("Iterations", fontsize = 10)
                axs[ax, ay].legend(fontsize = "small", ncol = 2, loc = "upper left")
                axs[ax, ay].set_xticks([])
            fig.tight_layout()
            fig.subplots_adjust(top = 0.91)
            fig.suptitle(
                "Dataset: " + str(folder) + "\nArchitecture: " + str(architecture),
                horizontalalignment = "left",
                x = 0.03,
                fontsize = 15,
                weight = "bold"
            )
            plt.savefig("/home/lucasmessias/MsC-Project/framework/results/" + str(folder) + "/active-results-" + str(architecture).replace(".", "_") + ".png")