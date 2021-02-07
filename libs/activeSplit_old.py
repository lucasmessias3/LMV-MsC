from oauth2client.service_account import ServiceAccountCredentials
from libs.activeSelectionLib import activeSelectionLib
from libs.utils import utils
import pandas as pd
import numpy as np
import gspread
import time

# Function to read CSV's
def readCSV(fileName):
    # To measure the time
    t = time.time()
    # Read csv into Pandas
    df = pd.read_csv("/home/lucasmessias/features/" + fileName, header = None).iloc[:, 1:]
    print('===' * 30)
    print('')
    print("CSV carregado com sucesso!")
    print("")
    print('Número de Features: {}'.format(len(df.iloc[0, :].values) - 1))
    print('Número Total de Amostras: {}'.format(len(df)))
    print('')
    print('Dividindo os dados em 80% para Treinamento e 20% para Teste..')
    # Split data into train and test 80/20 per class
    X_train, X_test, Y_train, Y_test = utilsObject.splitData(df)
    print("Divisão concluída: {} Amostras de Treino - {} Amostras de Teste".format(len(X_train), len(X_test)))
    print("")
    # Counts the unique samples of each class
    utilsObject.countUniqueLabels(Y_train, Y_test, "normalSplit")
    print("")
    print('Arquivo processado com sucesso! (Tempo de execucao: {})'.format(time.time() - t))
    print("")
    print('===' * 30)
    return X_train, X_test, Y_train, Y_test

# Function to split samples by the Active Selection
def activeSplit(technique, X_train, Y_train, numCluster, numNeighbors):
    # To measure the time
    t = time.time()
    print("")
    print("Dividindo amostras de treinamento pela Seleção Ativa utilizando o Método {}..".format(activeSelectionTechniques[technique]))
    print("")
    # Implementation of the First technique "Random Selection"
    if technique == "rand":
        # Split the Train data into 2 anothers subsets (supervised and unsupervised)
        sup_X, unsup_X, sup_Y, unsup_Y = activeSelectionLibObject.randomSelection(X_train, Y_train)
    # Implementation of the Second technique "K-Neighboors Samples Distance"
    elif technique == "border":
        # Split the Train data into 2 anothers subsets (supervised and unsupervised)
        sup_X, unsup_X, sup_Y, unsup_Y = activeSelectionLibObject.activeNeighbors(X_train, Y_train, int(numCluster), int(numNeighbors))
    else:
        print("Erro: Técnica de Seleção Ativa não encontrada")
        exit(0)
    print("Divisão concluída: {} Amostras Supervised - {} Amostras Unsupervised".format(len(sup_Y), len(unsup_Y)))
    print("")
    # Counts the unique samples of each class
    utilsObject.countUniqueLabels(sup_Y, unsup_Y, "activeSplit")
    print("")
    print('Divisão realizada com sucesso! (Tempo de execucao: {})'.format(time.time() - t))
    print("")
    print('===' * 30)
    return sup_X, unsup_X, sup_Y, unsup_Y

# ============================== PIPELINE ==============================

# This part is to get credentials, authorize the access at the results sheet and update the results
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('/home/lucasmessias/MsC-Project/framework/tools/googleSheet-API-Key.json', scope)
gc = gspread.authorize(credentials)
wks = gc.open("activeSplit").worksheet("Results")

# Define variables
activeSelectionTechniques = {"rand" : "Random Selection", "border" : "Border Samples Selection"}
# Build Objects
utilsObject = utils()
activeSelectionLibObject = activeSelectionLib()
# Data Input
dataInput = [
    # ["larvae/larvas_celso.csv", "rand", "0", "0"],
    # ["larvae/larvas_celso.csv", "border", "4", "4"],
    # ["larvae/inception_v3.csv", "rand", "0", "0"],
    # ["larvae/inception_v3.csv", "border", "4", "4"],
    # ["larvae/inception_resnet_v2.csv", "rand", "0", "0"],
    # ["larvae/inception_resnet_v2.csv", "border", "4", "4"],
    # ["eggs/ovos_celso.csv", "rand", "0", "0"],
    # ["eggs/ovos_celso.csv", "border", "9", "4"],
    # ["eggs/inception_v3.csv", "rand", "0", "0"],
    # ["eggs/inception_v3.csv", "border", "9", "3"],
    # ["eggs/inception_resnet_v2.csv", "rand", "0", "0"],
    # ["eggs/inception_resnet_v2.csv", "border", "9", "3"],
    # ["protozoan/proto_celso.csv", "rand", "0", "0"],
    # ["protozoan/proto_celso.csv", "border", "7", "3"],
    ["protozoan/inception_v3.csv", "rand", "0", "0"],
    ["protozoan/inception_v3.csv", "border", "7", "2"],
    # ["protozoan/inception_resnet_v2.csv", "rand", "0", "0"],
    # ["protozoan/inception_resnet_v2.csv", "border", "7", "2"],
    # ["all-15/inception_v3.csv", "rand", "0", "0"],
    # ["all-15/inception_v3.csv", "border", "15", "3"],
    # ["all-15/inception_resnet_v2.csv", "rand", "0", "0"],
    # ["all-15/inception_resnet_v2.csv", "border", "15", "3"],
    # ["all-16/inception_v3.csv", "rand", "0", "0"],
    # ["all-16/inception_v3.csv", "border", "16", "2"],
    # ["all-16/inception_resnet_v2.csv", "rand", "0", "0"],
    # ["all-16/inception_resnet_v2.csv", "border", "16", "2"],
]
# To control rows in spreadsheet
rowNumber = 7
# Loop for go through Data Input
for i in dataInput:
    # Read CSV
    X_train, X_test, Y_train, Y_test = readCSV(i[0])
    wks.update('A' + str(rowNumber), i[0])
    # Select and Save Samples Actively
    if i[1] == "rand":
        t = time.time()
        hard_X, easy_X, hard_Y, easy_Y = activeSplit(i[1], X_train, Y_train, i[2], i[3])
        timeElapsed = time.time() - t
        for name, df in zip(["hard_X", "easy_X", "hard_Y", "easy_Y", "test_X", "test_Y"],
                                                        [hard_X, easy_X, hard_Y, easy_Y, X_test, Y_test]):
            df.to_csv(str("/home/lucasmessias/MsC-Project/framework/data/" +
            i[0].split("/")[0] + "/" + i[0].split("/")[1].split(".")[0] + "/" +
            i[1] + "/" + str(name) + ".csv"))
        wks.update('D' + str(rowNumber), round(timeElapsed, 3))
        wks.update('G' + str(rowNumber), "---")
        wks.update('J' + str(rowNumber), "---")
        time.sleep(2)
        wks.update('M' + str(rowNumber), "---")
        wks.update('P' + str(rowNumber), len(hard_Y))
        wks.update('S' + str(rowNumber), len(easy_Y))
    if i[1] == "border":
        t = time.time()
        hard_X, easy_X, hard_Y, easy_Y = activeSplit(i[1], X_train, Y_train, i[2], i[3])
        timeElapsed = time.time() - t
        for name, df in zip(["hard_X", "easy_X", "hard_Y", "easy_Y", "test_X", "test_Y"],
                                                        [hard_X, easy_X, hard_Y, easy_Y, X_test, Y_test]):
            df.to_csv(str("/home/lucasmessias/MsC-Project/framework/data/" +
            i[0].split("/")[0] + "/" + i[0].split("/")[1].split(".")[0] + "/" +
            i[1] + "/" + str(name) + ".csv"))
        wks.update('D' + str(rowNumber), "---")
        wks.update('G' + str(rowNumber), round(timeElapsed, 3))
        wks.update('J' + str(rowNumber), i[2])
        time.sleep(2)
        wks.update('M' + str(rowNumber), i[3])
        wks.update('P' + str(rowNumber), len(hard_Y))
        wks.update('S' + str(rowNumber), len(easy_Y))
    rowNumber += 1