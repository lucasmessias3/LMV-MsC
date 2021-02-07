from tkinter import *
from utils import utils
from activeSelectionLib import activeSelectionLib
from activeLearningLib import activeLearningLib
from classifiersLib import classifiersLib
from semiSupervisedLib import semiSupervisedLib
from plotLib import plotLib
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

class Application: 

    # Building the object's useds in this code
    utilsObject = utils()
    activeSelectionLibObject = activeSelectionLib()
    activeLearningLibObject = activeLearningLib()
    classifiersLibObject = classifiersLib()
    plotLibObject = plotLib()
    semiSupervisedLibObject = semiSupervisedLib()

    def __init__(self, master = None):

        # Put a name in the Main Window
        master.wm_title("Desenvolvido por Lucas Messias")

        # Creating the Program Title
        label_title = Label(
            master,
            text = 'Framework Mestrado',
            font = 'Helvetica -30 bold'
        )
        label_title.grid(row = 0, column = 1)

        # Button to load arffs
        button_arff = Button(
            master,
            text = 'Carregar arquivo ARFF',
            command = self.readArff,
            font = 'Helvetica -18 bold',
            foreground = 'white',
            background = 'black'
        )
        button_arff.grid(row = 1, column = 1)

        # Option Menu to select the Active Selection Technique
        label_option_menu_1 = Label(
            master,
            text = 'Escolha o método de Seleção Ativa das Amostras: ', 
            font = 'Helvetica -18 bold'
        )
        label_option_menu_1.grid(row = 2, column = 1)

        optionMenu_ActiveSelecion = OptionMenu(
            master,
            selectedActiveSelection,
            *activeSelectionList
        )
        optionMenu_ActiveSelecion.grid(row = 3, column = 1)

        # Button to Select samples by Active Selection Technique
        button_ActiveSelecion = Button(
            master,
            text = 'Dividir Amostras de forma Ativa',
            command = self.selectActiveSamples,
            font = 'Helvetica -18 bold',
            foreground = 'white',
            background = 'black'
        )
        button_ActiveSelecion.grid(row = 4, column = 1)

        # Option Menu to select the Active Learning Technique
        label_option_menu_2 = Label(
            master,
            text = 'Escolha o método de Aprendizado Ativo: ', 
            font = 'Helvetica -18 bold'
        )
        label_option_menu_2.grid(row = 5, column = 1)

        optionMenu_ActiveLearning = OptionMenu(
            master,
            selectedActiveLearning,
            *activeLearningList
        )
        optionMenu_ActiveLearning.grid(row = 6, column = 1)

        # Option Menu to select the Semi-Supervised Learning Technique
        label_option_menu_3 = Label(
            master,
            text = 'Escolha o método de Aprendizado Semi Supervisionado: ', 
            font = 'Helvetica -18 bold'
        )
        label_option_menu_3.grid(row = 7, column = 1)

        optionMenu_SemiSupervisedLearning = OptionMenu(
            master,
            selectedSemiSupervised,
            *semiSupervisedList
        )
        optionMenu_SemiSupervisedLearning.grid(row = 8, column = 1)

        # SpinBox to select K Value
        label_SpinBox_K = Label(
            master,
            text = 'Selecione o valor de K-amostras por iteração: ', 
            font = 'Helvetica -18 bold'
        )
        label_SpinBox_K.grid(row = 9, column = 1)

        self.spinBox_K = Spinbox(
            master,
            from_ = 4,
            to = 100
        )
        self.spinBox_K.grid(row = 10, column = 1)

        # SpinBox for selecting the number of iterations that are performed in the Pipeline
        label_SpinBox = Label(
            master,
            text = 'Selecione o número de iterações que o Pipeline rodará: ', 
            font = 'Helvetica -18 bold'
        )
        label_SpinBox.grid(row = 11, column = 1)
        self.spinbox_PipelineRuns = Spinbox(
            master,
            from_ = 2,
            to = 100
        )
        self.spinbox_PipelineRuns.grid(row = 12, column = 1)

        # Button to Run Pipeline
        button_RunPipeline = Button(
            master,
            text = 'Rodar Pipeline',
            command = self.runPipeline,
            font = 'Helvetica -18 bold',
            foreground = 'white',
            background = 'black'
        )
        button_RunPipeline.grid(row = 13, column = 1)

    # Function to read Arffs
    def readArff(self):
        # To measure the time
        t = time.time()
        # Read the Arff input
        df = self.utilsObject.readArff()
        print('===' * 30)
        print('')
        print("ARFF carregado com sucesso!")
        print("")
        print('Número de Features: {}'.format(len(df.iloc[0, :].values) - 1))
        print('Número Total de Amostras: {}'.format(len(df)))
        print('')
        print('Dividindo os dados em 80% para Treinamento e 20% para Teste..')
        # Create global variables
        global X_train, X_test, Y_train, Y_test
        # Split data into train and test 80/20 per class
        X_train, X_test, Y_train, Y_test = self.utilsObject.splitData(df)
        print("Divisão concluída: {} Amostras de Treino - {} Amostras de Teste".format(len(X_train), len(X_test)))
        print("")
        print("Disposição das amostras no Conjunto de Treinamento:")
        # Counts the unique samples of each class from the training data
        unique, counts = np.unique(Y_train.values, return_counts = True)
        for i in zip(unique, counts):
            print("      Número de amostras da Classe {} : {}".format(i[0], i[1]))
        print("Disposição das amostras no Conjunto de Teste:")
        # Counts the unique samples of each class from the test data
        unique, counts = np.unique(Y_test.values, return_counts = True)
        for i in zip(unique, counts):
            print("      Número de amostras da Classe {} : {}".format(i[0], i[1]))
        print("")
        print('Arquivo processado com sucesso! (Tempo de execucao: {})'.format(time.time() - t))
        print("")
        print('===' * 30)

    # Function to split samples by the Active Selection
    def selectActiveSamples(self):
        # To measure the time
        t = time.time()
        print("")
        print("Dividindo amostras de treinamento pela Seleção Ativa utilizando o Método {}..".format(selectedActiveSelection.get()))
        print("")
        # Define the global variables for the Training Set
        global sup_X, unsup_X, sup_Y, unsup_Y
        # Implementation of the First technique "Random Selection"
        if selectedActiveSelection.get() == "Random Selection":
            # Split the Train data into 2 anothers subsets (supervised and unsupervised)
            sup_X, unsup_X, sup_Y, unsup_Y = self.activeSelectionLibObject.randomSelection(X_train, Y_train)
        # Implementation of the Second technique "K-Neighboors Samples Distance"
        if selectedActiveSelection.get() == "ActiveNeighbors":
            # Split the Train data into 2 anothers subsets (supervised and unsupervised)
            sup_X, unsup_X, sup_Y, unsup_Y = self.activeSelectionLibObject.activeNeighbors(X_train, Y_train)
        print("Divisão concluída: {} Amostras Supervised - {} Amostras Unsupervised".format(len(sup_Y), len(unsup_Y)))
        print("")
        print("Disposição das amostras no Conjunto Supervised:")
        # Counts the unique samples of each class from the supervised data
        unique, counts = np.unique(sup_Y.values, return_counts = True)
        for i in zip(unique, counts):
            print("      Número de amostras da Classe {} : {}".format(i[0], i[1]))
        print("Disposição das amostras no Conjunto Unsupervised:")
        # Counts the unique samples of each class from the unsupervised data
        unique, counts = np.unique(unsup_Y.values, return_counts = True)
        for i in zip(unique, counts):
            print("      Número de amostras da Classe {} : {}".format(i[0], i[1]))
        print("")
        print('Divisão realizada com sucesso! (Tempo de execucao: {})'.format(time.time() - t))
        print("")
        print('===' * 30)

    # Function to Train Classifiers
    def trainClassifiers(self, accList_all, f1ScoreList_all, precisionList_all, recallList_all, labeledData_SS, X_test, Y_test):
        accList = []; f1ScoreList = []; precisionList = []; recallList = []
        # Train classifiers with the labeledData and classify test Data
        # Save accuracy, f1-score, precision and recall in some vectors to after build a DataFrame
        acc, f1Score, precision, recall = self.classifiersLibObject.gaussianNB(labeledData_SS, X_test, Y_test)
        accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
        acc, f1Score, precision, recall = self.classifiersLibObject.logisticRegression(labeledData_SS, X_test, Y_test)
        accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
        acc, f1Score, precision, recall = self.classifiersLibObject.decisionTree(labeledData_SS, X_test, Y_test)
        accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
        acc, f1Score, precision, recall = self.classifiersLibObject.knn(labeledData_SS, X_test, Y_test)
        accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
        acc, f1Score, precision, recall = self.classifiersLibObject.LinearDiscriminantAnalysis(labeledData_SS, X_test, Y_test)
        accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
        acc, f1Score, precision, recall = self.classifiersLibObject.supportVectorMachine(labeledData_SS, X_test, Y_test)
        accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
        acc, f1Score, precision, recall = self.classifiersLibObject.RandomForest(labeledData_SS, X_test, Y_test)
        accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
        acc, f1Score, precision, recall = self.classifiersLibObject.neuralNet(labeledData_SS, X_test, Y_test)
        accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
        acc, f1Score, precision, recall = self.classifiersLibObject.opf(labeledData_SS, X_test, Y_test)
        accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)

        # Save the results from each iteration in this list's
        accList_all.append(accList); f1ScoreList_all.append(f1ScoreList); precisionList_all.append(precisionList); recallList_all.append(recallList)
        return accList_all, f1ScoreList_all, precisionList_all, recallList_all


    # Function that control the Pipeline Runs
    def runPipeline(self):
        # List to save RDS Lists
        rdsLists = []
        # List to get the TrueLabels of rdsLists
        truLabel_rdsLists = []
        # Control the first run of RDS
        rdsFirst = True
        # To measure the time
        t = time.time()
        # Get the global variables for Supervised Data and Test Data
        global sup_X, sup_Y, X_test, Y_test
        # Creates the variable responsible for storing the data labeled by the specialist
        global labeledData
        labeledData = pd.DataFrame(columns = pd.concat([sup_X, sup_Y], axis = 1).columns.tolist())
        # Create vectors to save classfiers result
        accList_all = []
        f1ScoreList_all = []
        precisionList_all = []
        recallList_all = []
        print("")
        print("Escolhendo amostras para o especialista por meio do método {}..".format(selectedActiveLearning.get()))
        # # Loop to do the iterations with the selected Technique
        for i in range(0, int(self.spinbox_PipelineRuns.get())):
            print("")
            print("Começando a Iteração de número {}".format(i + 1))
            print('--' * 30)
            print("")
            # If to get the Active Learning technique selected
            if selectedActiveLearning.get() == "Random Selection":
                # This function select data and pass to specialist, returns labeled Data and Supervised Data updated
                auxDF, sup_X, sup_Y = self.activeLearningLibObject.randomActiveLearning(int(self.spinBox_K.get()), sup_X, sup_Y)
                labeledData = labeledData.append(auxDF)
                print("Número de amostras no Conjunto rotulado pelo Especialista: {}".format(len(labeledData.iloc[:, -1:].values)))
                print("Amostras selecionadas ativamente e rotuladas pelo especialista!")
                print("")
                print("Propagando labels para todo Conjunto Supervised com o método {}..".format(selectedSemiSupervised.get()))
                # IF to get the Semi Supervised technique selected
                if selectedSemiSupervised.get() == "PseudoLabeling":
                    # Propagate labels to Supervised Data
                    # Predict the sup_X, concat with the data and append the labeledData by specialist to a full sup_X Labeled and PseudoLabeled
                    labeledData_SS = pd.DataFrame(columns = sup_X.columns.tolist())
                    labeledData_SS = pd.concat([sup_X.reset_index(drop = True), self.semiSupervisedLibObject.pseudoLabeling(labeledData, sup_X, "Logistic Regression")], axis = 1)
                    labeledData_SS = labeledData_SS.append(labeledData)
                    print("Número de amostras no Conjunto que será utilizado para treinar o Classificador: {}".format(len(labeledData_SS)))
                    print("")
                if selectedSemiSupervised.get() == "OPFSemi":
                    # Propagate labels to Supervised Data
                    # Predict the sup_X, concat with the data and append the labeledData by specialist to a full sup_X Labeled and PseudoLabeled
                    labeledData_SS = pd.DataFrame(columns = sup_X.columns.tolist())
                    labeledData_SS = pd.concat([sup_X.reset_index(drop = True), self.semiSupervisedLibObject.opfSemi(labeledData, sup_X, sup_Y)], axis = 1)
                    labeledData_SS = labeledData_SS.append(labeledData)
                    print("Número de amostras no Conjunto que será utilizado para treinar o Classificador: {}".format(len(labeledData_SS)))
                    print("")
                print("Treinando os Classificadores e classificando o Conjunto de Testes..")
                print("")
                # Save the results from each iteration in this list's
                accList_all, f1ScoreList_all, precisionList_all, recallList_all = self.trainClassifiers(accList_all, f1ScoreList_all, precisionList_all, recallList_all, labeledData_SS, X_test, Y_test)
            # If to get the Active Learning technique selected
            if selectedActiveLearning.get() == "RDS":
                auxDF, sup_X, sup_Y, rdsLists, truLabel_rdsLists = self.activeLearningLibObject.rds(int(self.spinBox_K.get()), sup_X, sup_Y, rdsFirst, rdsLists, labeledData, truLabel_rdsLists)
                rdsFirst = False
                labeledData = labeledData.append(auxDF)
                print("Número de amostras no Conjunto rotulado pelo Especialista: {}".format(len(labeledData.iloc[:, -1:].values)))
                print("Amostras selecionadas ativamente e rotuladas pelo especialista!")
                print("")
                print("Propagando labels para todo Conjunto Supervised com o método {}..".format(selectedSemiSupervised.get()))
                # IF to get the Semi Supervised technique selected
                if selectedSemiSupervised.get() == "PseudoLabeling":
                    # Propagate labels to Supervised Data
                    # Predict the sup_X, concat with the data and append the labeledData by specialist to a full sup_X Labeled and PseudoLabeled
                    labeledData_SS = pd.DataFrame(columns = sup_X.columns.tolist())
                    labeledData_SS = pd.concat([sup_X.reset_index(drop = True), self.semiSupervisedLibObject.pseudoLabeling(labeledData, sup_X, "Logistic Regression")], axis = 1)
                    labeledData_SS = labeledData_SS.append(labeledData)
                    print("Número de amostras no Conjunto que será utilizado para treinar o Classificador: {}".format(len(labeledData_SS)))
                    print("")
                if selectedSemiSupervised.get() == "OPFSemi":
                    # Propagate labels to Supervised Data
                    # Predict the sup_X, concat with the data and append the labeledData by specialist to a full sup_X Labeled and PseudoLabeled
                    labeledData_SS = pd.DataFrame(columns = sup_X.columns.tolist())
                    labeledData_SS = pd.concat([sup_X.reset_index(drop = True), self.semiSupervisedLibObject.opfSemi(labeledData, sup_X, sup_Y)], axis = 1)
                    labeledData_SS = labeledData_SS.append(labeledData)
                    print("Número de amostras no Conjunto que será utilizado para treinar o Classificador: {}".format(len(labeledData_SS)))
                    print("")
                print("Treinando os Classificadores e classificando o Conjunto de Testes..")
                print("")
                # Save the results from each iteration in this list's
                accList_all, f1ScoreList_all, precisionList_all, recallList_all = self.trainClassifiers(accList_all, f1ScoreList_all, precisionList_all, recallList_all, labeledData_SS, X_test, Y_test)
        print("")
        print('===' * 30)
        print("")
        print('Pipeline finalizado com sucesso! (Tempo de execucao: {})'.format(time.time() - t))
        print("")
        print("Plotando resultados..")
        print("")
        # Convert lists into DF's with the classifiers name as column's
        accDF, f1DF, preDF, recDF = self.plotLibObject.listsToPlotDF(accList_all, f1ScoreList_all, precisionList_all, recallList_all)
        # Plot results
        self.plotLibObject.plotResults(accDF, f1DF, preDF, recDF)
        print('===' * 30)

# -------------------- MAIN FUNCTION --------------------
# Instantiate the TK() Classe allowing widgets to be used
root = Tk()

# List with possibles Active Selection Techniques
activeSelectionList = ['Random Selection', 'ActiveNeighbors', 'Técnica 2']
selectedActiveSelection = StringVar()
# Set the index 0 for default Active Selection
selectedActiveSelection.set(activeSelectionList[0])

# List with possibles Active Learning Techniques
activeLearningList = ['Random Selection', 'RDS', 'Técnica 2']
selectedActiveLearning = StringVar()
# Set the index 0 for default Active Learning
selectedActiveLearning.set(activeLearningList[1])

# List with possibles Semi-Supervised Techniques
semiSupervisedList = ['PseudoLabeling', 'OPFSemi', 'Técnica 2']
selectedSemiSupervised = StringVar()
# Set the index 0 for default Semi-Supervised Learning
selectedSemiSupervised.set(semiSupervisedList[1])

# Pass the root to class Application and start the Main Loop
Application(root)
root.mainloop()