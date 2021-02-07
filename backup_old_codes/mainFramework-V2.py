from oauth2client.service_account import ServiceAccountCredentials
from activeSelectionLib import activeSelectionLib
from activeLearningLib import activeLearningLib
from semiSupervisedLib import semiSupervisedLib
from classifiersLib import classifiersLib
from plotLib import plotLib
from utils import utils
import pandas as pd
import numpy as np
import gspread
import arff
import time
import sys

# Function to read Arffs
def readArff(fileName):
    # First Clean all Rows
    table = [["", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", ""], 
            ["", "", "", "", "", "", "", "", "", ""], 
            ["", "", "", "", "", "", "", "", "", ""], 
            ["", "", "", "", "", "", "", "", "", ""], 
            ["", "", "", "", "", "", "", "", "", ""], 
            ["", "", "", "", "", "", "", "", "", ""]]
    wks.update('D3:D5', [[""], [""], [""]])
    wks.update('J3:J5', [[""], [""], [""]])
    wks.update('P3:P5', [[""], [""], [""]])
    wks.update('T3:T5', [[""], [""], [""]])
    wks.update('D9', table)
    wks.update('D39', table)
    wks.update('Q9', table)
    wks.update('Q39', table)
    # To measure the time
    t = time.time()
    # Read the Arff input
    data = arff.load(open(fileName))
    # Update the dataset name in the Google Sheet
    wks.update('D3', fileName)
    df = pd.DataFrame(data['data']).iloc[:, 1:]
    print('===' * 30)
    print('')
    print("ARFF carregado com sucesso!")
    print("")
    print('Número de Features: {}'.format(len(df.iloc[0, :].values) - 1))
    print('Número Total de Amostras: {}'.format(len(df)))
    print('')
    # Update the Features and Samples Numbers name in the Google Sheet
    wks.update('D4', int(len(df.iloc[0, :].values) - 1))
    wks.update('D5', int(len(df)))
    print('Dividindo os dados em 80% para Treinamento e 20% para Teste..')
    # Create global variables
    global X_train, X_test, Y_train, Y_test
    # Split data into train and test 80/20 per class
    X_train, X_test, Y_train, Y_test = utilsObject.splitData(df)
    print("Divisão concluída: {} Amostras de Treino - {} Amostras de Teste".format(len(X_train), len(X_test)))
    print("")
    # Counts the unique samples of each class
    utilsObject.countUniqueLabels(Y_train, Y_test)
    print("")
    print('Arquivo processado com sucesso! (Tempo de execucao: {})'.format(time.time() - t))
    print("")
    print('===' * 30)

# Function to split samples by the Active Selection
def selectActiveSamples(technique):
    # To measure the time
    t = time.time()
    print("")
    print("Dividindo amostras de treinamento pela Seleção Ativa utilizando o Método {}..".format(activeSelecionTechniques[technique]))
    print("")
    # Update the Active Selection Method name in the Google Sheet
    wks.update('J3', activeSelecionTechniques[technique])
    # Define the global variables for the Training Set
    global sup_X, unsup_X, sup_Y, unsup_Y
    # Implementation of the First technique "Random Selection"
    if technique == "RS":
        # Split the Train data into 2 anothers subsets (supervised and unsupervised)
        sup_X, unsup_X, sup_Y, unsup_Y = activeSelectionLibObject.randomSelection(X_train, Y_train)
    # Implementation of the Second technique "K-Neighboors Samples Distance"
    elif technique == "KN":
        # Split the Train data into 2 anothers subsets (supervised and unsupervised)
        sup_X, unsup_X, sup_Y, unsup_Y = activeSelectionLibObject.activeNeighbors(X_train, Y_train)
    else:
        print("Erro: Técnica de Seleção Ativa não encontrada")
        exit(0)
    print("Divisão concluída: {} Amostras Supervised - {} Amostras Unsupervised".format(len(sup_Y), len(unsup_Y)))
    print("")
    # Counts the unique samples of each class
    utilsObject.countUniqueLabels(sup_Y, unsup_Y)
    print("")
    print('Divisão realizada com sucesso! (Tempo de execucao: {})'.format(time.time() - t))
    print("")
    print('===' * 30)

# Function to Train Classifiers
def trainClassifiers(accList_all, f1ScoreList_all, precisionList_all, recallList_all, labeledData_SS, X_test, Y_test):
    accList = []; f1ScoreList = []; precisionList = []; recallList = []
    # Train classifiers with the labeledData and classify test Data
    # Save accuracy, f1-score, precision and recall in some vectors to after build a DataFrame
    acc, f1Score, precision, recall = classifiersLibObject.gaussianNB(labeledData_SS, X_test, Y_test)
    accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
    acc, f1Score, precision, recall = classifiersLibObject.logisticRegression(labeledData_SS, X_test, Y_test)
    accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
    acc, f1Score, precision, recall = classifiersLibObject.decisionTree(labeledData_SS, X_test, Y_test)
    accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
    acc, f1Score, precision, recall = classifiersLibObject.knn(labeledData_SS, X_test, Y_test)
    accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
    acc, f1Score, precision, recall = classifiersLibObject.LinearDiscriminantAnalysis(labeledData_SS, X_test, Y_test)
    accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
    acc, f1Score, precision, recall = classifiersLibObject.supportVectorMachine(labeledData_SS, X_test, Y_test)
    accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
    acc, f1Score, precision, recall = classifiersLibObject.RandomForest(labeledData_SS, X_test, Y_test)
    accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
    acc, f1Score, precision, recall = classifiersLibObject.neuralNet(labeledData_SS, X_test, Y_test)
    accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
    acc, f1Score, precision, recall = classifiersLibObject.opf(labeledData_SS, X_test, Y_test)
    accList.append(acc); f1ScoreList.append(f1Score) ; precisionList.append(precision) ; recallList.append(recall)
    # Save the results from each iteration in this list's
    accList_all.append(accList); f1ScoreList_all.append(f1ScoreList); precisionList_all.append(precisionList); recallList_all.append(recallList)
    return accList_all, f1ScoreList_all, precisionList_all, recallList_all

# Function that control the Pipeline Runs
def runPipeline(activeLearning, semiSupervised, valueOfK, iterationsNumber, rdsClassifier):
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
    print("Escolhendo amostras para o especialista por meio do método {}..".format(activeLearningTechniques[activeLearning]))
    # Update the Active Learning Method name in the Google Sheet
    wks.update('J4', activeLearningTechniques[activeLearning])
    # Update the RDS Classifier name in the Google Sheet
    wks.update('T3', rdsClassifier)
    # # Loop to do the iterations with the selected Technique
    for i in range(0, int(iterationsNumber)):
        print("")
        print("Começando a Iteração de número {}".format(i + 1))
        print('--' * 30)
        print("")
        # If to get the Active Learning technique selected
        if activeLearning == "RS":
            # This function select data and pass to specialist, returns labeled Data and Supervised Data updated
            auxDF, sup_X, sup_Y = activeLearningLibObject.randomActiveLearning(int(valueOfK), sup_X, sup_Y)
            labeledData = labeledData.append(auxDF)
            print("Número de amostras no Conjunto rotulado pelo Especialista: {}".format(len(labeledData.iloc[:, -1:].values)))
            print("Amostras selecionadas ativamente e rotuladas pelo especialista!")
            print("")
            print("Propagando labels para todo Conjunto Supervised com o método {}..".format(semiSupervisedTechniques[semiSupervised]))
            # Update the Active Learning Method name in the Google Sheet
            wks.update('J5', semiSupervisedTechniques[semiSupervised])
            # # IF to get the Semi Supervised technique selected
            # if semiSupervised == "PL":
            #     # Propagate labels to Supervised Data
            #     # Predict the sup_X, concat with the data and append the labeledData by specialist to a full sup_X Labeled and PseudoLabeled
            #     labeledData_SS = pd.DataFrame(columns = sup_X.columns.tolist())
            #     labeledData_SS = pd.concat([sup_X.reset_index(drop = True), semiSupervisedLibObject.pseudoLabeling(labeledData, sup_X, "Logistic Regression")], axis = 1)
            #     labeledData_SS = labeledData_SS.append(labeledData)
            #     print("Número de amostras no Conjunto que será utilizado para treinar o Classificador: {}".format(len(labeledData_SS)))
            #     print("")
            # if semiSupervised == "OS":
            #     # Propagate labels to Supervised Data
            #     # Predict the sup_X, concat with the data and append the labeledData by specialist to a full sup_X Labeled and PseudoLabeled
            #     labeledData_SS = pd.DataFrame(columns = sup_X.columns.tolist())
            #     labeledData_SS = pd.concat([sup_X.reset_index(drop = True), semiSupervisedLibObject.opfSemi(labeledData, sup_X, sup_Y)], axis = 1)
            #     labeledData_SS = labeledData_SS.append(labeledData)
            #     print("Número de amostras no Conjunto que será utilizado para treinar o Classificador: {}".format(len(labeledData_SS)))
            #     print("")
            print("Treinando os Classificadores e classificando o Conjunto de Testes..")
            print("")
            # Save the results from each iteration in this list's
            accList_all, f1ScoreList_all, precisionList_all, recallList_all = trainClassifiers(accList_all, f1ScoreList_all, precisionList_all, recallList_all, labeledData, X_test, Y_test)
        # If to get the Active Learning technique selected
        if activeLearning == "RDS":
            auxDF, sup_X, sup_Y, rdsLists, truLabel_rdsLists = activeLearningLibObject.rds(int(valueOfK), sup_X, sup_Y, rdsFirst, rdsLists, labeledData, truLabel_rdsLists, rdsClassifier)
            rdsFirst = False
            labeledData = labeledData.append(auxDF)
            print("Número de amostras no Conjunto rotulado pelo Especialista: {}".format(len(labeledData.iloc[:, -1:].values)))
            print("Amostras selecionadas ativamente e rotuladas pelo especialista!")
            print("")
            print("Propagando labels para todo Conjunto Supervised com o método {}..".format(semiSupervisedTechniques[semiSupervised]))
            # Update the Active Learning Method name in the Google Sheet
            wks.update('J5', semiSupervisedTechniques[semiSupervised])
            # # IF to get the Semi Supervised technique selected
            # if semiSupervised == "PL":
            #     # Propagate labels to Supervised Data
            #     # Predict the sup_X, concat with the data and append the labeledData by specialist to a full sup_X Labeled and PseudoLabeled
            #     labeledData_SS = pd.DataFrame(columns = sup_X.columns.tolist())
            #     labeledData_SS = pd.concat([sup_X.reset_index(drop = True), semiSupervisedLibObject.pseudoLabeling(labeledData, sup_X, "Logistic Regression")], axis = 1)
            #     labeledData_SS = labeledData_SS.append(labeledData)
            #     print("Número de amostras no Conjunto que será utilizado para treinar o Classificador: {}".format(len(labeledData_SS)))
            #     print("")
            # if semiSupervised == "OS":
            #     # Propagate labels to Supervised Data
            #     # Predict the sup_X, concat with the data and append the labeledData by specialist to a full sup_X Labeled and PseudoLabeled
            #     labeledData_SS = pd.DataFrame(columns = sup_X.columns.tolist())
            #     labeledData_SS = pd.concat([sup_X.reset_index(drop = True), semiSupervisedLibObject.opfSemi(labeledData, sup_X, sup_Y)], axis = 1)
            #     labeledData_SS = labeledData_SS.append(labeledData)
            #     print("Número de amostras no Conjunto que será utilizado para treinar o Classificador: {}".format(len(labeledData_SS)))
            #     print("")
            print("Treinando os Classificadores e classificando o Conjunto de Testes..")
            print("")
            # Save the results from each iteration in this list's
            accList_all, f1ScoreList_all, precisionList_all, recallList_all = trainClassifiers(accList_all, f1ScoreList_all, precisionList_all, recallList_all, labeledData, X_test, Y_test)
    print("")
    print('===' * 30)
    print("")
    print('Pipeline finalizado com sucesso! (Tempo de execucao: {})'.format(time.time() - t))
    print("")
    print("Preenchendo planilha de resultados..")
    print("")
    wks.update('P3', int(valueOfK))
    wks.update('P4', int(iterationsNumber))
    # Convert lists into DF's with the classifiers name as column's
    accDF, f1DF, preDF, recDF = plotLibObject.listsToPlotDF(accList_all, f1ScoreList_all, precisionList_all, recallList_all)
    # Create 2 variables to help write results in Spreadsheet
    # cellValue 10 is the number of starter cell in Acuraccy Table from SpreadSheet
    # control is to help go through columns
    alphabet = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    cellValue = 9
    control = True
    for index, row in accDF.iterrows():
        for i in row:
            aux = index
            if control == True:
                aux2 = index
                control = False
            if aux2 != aux:
                aux2 = aux
                control = True
                cellValue = 9
            # Update the Accuracy in the Google Sheet
            time.sleep(1)
            wks.update(str(alphabet[index] + str(cellValue)), round(i, 5))
            cellValue += 1
    cellValue = 39
    control = True
    for index, row in f1DF.iterrows():
        for i in row:
            aux = index
            if control == True:
                aux2 = index
                control = False
            if aux2 != aux:
                aux2 = aux
                control = True
                cellValue = 39
            # Update the Accuracy in the Google Sheet
            time.sleep(1)
            wks.update(str(alphabet[index] + str(cellValue)), round(i, 5))
            cellValue += 1
    alphabet = ['Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    cellValue = 9
    control = True
    for index, row in preDF.iterrows():
        for i in row:
            aux = index
            if control == True:
                aux2 = index
                control = False
            if aux2 != aux:
                aux2 = aux
                control = True
                cellValue = 9
            # Update the Accuracy in the Google Sheet
            time.sleep(1)
            wks.update(str(alphabet[index] + str(cellValue)), round(i, 5))
            cellValue += 1
    cellValue = 39
    control = True
    for index, row in recDF.iterrows():
        for i in row:
            aux = index
            if control == True:
                aux2 = index
                control = False
            if aux2 != aux:
                aux2 = aux
                control = True
                cellValue = 39
            # Update the Accuracy in the Google Sheet
            time.sleep(1)
            wks.update(str(alphabet[index] + str(cellValue)), round(i, 5))
            cellValue += 1
    # Plot results
    # plotLibObject.plotResults(accDF, f1DF, preDF, recDF)
    print('===' * 30)

# This part is to get credentials, authorize the access at the results sheet and update the results
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('/home/messias/Dropbox/GitHub/MsC-Project/framework/tools/googleSheet-API-Key.json', scope)
gc = gspread.authorize(credentials)
wks = gc.open("LMV-MsC-Results").worksheet(sys.argv[7])

utilsObject = utils()
activeSelectionLibObject = activeSelectionLib()
activeLearningLibObject = activeLearningLib()
semiSupervisedLibObject = semiSupervisedLib()
classifiersLibObject = classifiersLib()
plotLibObject = plotLib()

activeSelecionTechniques = {"RS" : "Random Selection", "KN" : "K-Neighboors Samples Distance"}
activeLearningTechniques = {"RS" : "Random Selection", "RDS" : "Root Distance based Sampling"}
semiSupervisedTechniques = {"PL" : "Pseudo-Labeling", "OS" : "Optimun-Path Forest SEMI"}

start_time = time.time()
readArff(sys.argv[1])
selectActiveSamples(sys.argv[2])
runPipeline(sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[8] if len(sys.argv) == 9 else "null")
wks.update('P5', int(time.time() - start_time))

# Cenários Possíveis

# python mainFramework-V2.py <arff-file> <activeSelectionTechnique> <activeLearningTechnique> <semiSupervisedTechnique> <valueOfK> <iterationNumber> <SheetName> <RDSClassifier>

# <arff-file> - features/larvas_celso.arff, features/ovos_celso.arff, features/proto_celso.arff
# <activeSelectionTechnique> - RS (Random Selection), KN (K-Neighboors Samples Distance)
# <activeLearningTechnique> - RS (Random Selection), RDS (Root Distance base Sampling)
# <semiSupervisedTechnique> - PL (Pseudo-Labeling), OS (Optimun-Path Forest SEMI)
# <valueOfK> - Valor do K, exemplo: 4
# <iterationNumber> - Número de Iterações do programa, exemplo: 10
# <SheetName> - Nome da aba da planilha que será escrito os resultados
# <RDSClassifier> - Define o classificador utilizado no RDS, possíveis: "GaussianNB", "LogisticRegression", "k-NN", "LDA", "SVM", "RandomForest", "NeuralNet" ou "null" (caso não use)