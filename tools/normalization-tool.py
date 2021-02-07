from tkinter import *
from tkinter import filedialog
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import os, sys
import arff
import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

class Application:

    def __init__(self, master = None):

        # Put a name in the Main Window
        master.wm_title("Desenvolvido por Lucas Messias")
        # Creating the Program Title
        label_title = Label(
            master,
            text = 'Ferramenta para Normalização',
            font = 'Helvetica -30 bold')
        label_title.grid(row = 0, column = 1)
        # Choosing the standardization technique
        label_normalization = Label(
            master,
            text = 'Marque as técnicas de Normalizacao desejadas:', 
            font = 'Helvetica -18 bold')
        label_normalization.grid(row = 1, column = 1)
        norm1 = Checkbutton(master, text = "MinMaxScaler", variable = norm1_int).grid(row = 2, column = 1)
        norm2 = Checkbutton(master, text = "StandardScaler", variable = norm2_int).grid(row = 3, column = 1)
        norm3 = Checkbutton(master, text = "MaxAbsScaler", variable = norm3_int).grid(row = 4, column = 1)
        norm4 = Checkbutton(master, text = "RobustScaler", variable = norm4_int).grid(row = 5, column = 1)
        # Load Arff Folder
        button_arff = Button(
            master,
            text = 'Selecione um diretório e todos Arff\'s serão normalizados.',
            command = self.normalizeArffs,
            font = 'Helvetica -18 bold',
            foreground = 'white',
            background = 'black')
        button_arff.grid(row = 6, column = 1)

    def normalizeFunction(self, data, technique, folderName, arffs):
            t = time.time()
            # Choose Standardization Technique
            if technique == 'MinMaxScaler':
                scaler = MinMaxScaler()
            if technique == 'StandardScaler':
                scaler = StandardScaler()
            if technique == 'MaxAbsScaler':
                scaler = MaxAbsScaler()
            if technique == 'RobustScaler':
                scaler = RobustScaler()
            print("Convertendo para Normalização {}..".format(technique))
            print("Transformando em Dataframe..")
            # Convert data into DataFrame
            df = pd.DataFrame(data['data'])
            # Get features number
            length = df.iloc[0, :].values
            print("Número de Features: {}".format(len(length) - 1))
            # Create a label column
            labels = df.iloc[:, len(length) - 1].values
            # Save features without labels
            data_aux = df.iloc[:, 0:(len(length) - 1)].values
            # Normalize data
            data_normalized = scaler.fit_transform(data_aux)
            # Adding the labels to normalized data
            data_normalized = np.concatenate((data_normalized, np.vstack(labels)), axis = 1)
            # Replacing data with normalized samples
            data['data'] = data_normalized
            # Creating Folder if doesnt exists
            try:
                os.mkdir(folderName + "Normalized_Arffs", 755)
                print("Criando pasta onde será salvo os arquivos arffs..")
            except:
                print("Pasta já existente, apenas sobrescrevendo os arquivos arffs..")
            # Saving arff in text file
            print("Salvando arff..")
            newArffFile = open(folderName + "Normalized_Arffs/" + arffs[:-5] + "_" + technique + ".arff", "w")
            newArffFile.write(arff.dumps(data))
            newArffFile.close()
            print("Processo para o arquivo {} terminado. (Tempo de execução: {})".format(arffs, time.time() - t))
            print("")

    def normalizeArffs(self):
        print('===' * 30)
        print('')
        print('Selecionando diretório..')
        print('')
        folderName =  filedialog.askdirectory() + "/"
        print('Diretório selecionado: ' + folderName)
        print('')
        arffNames = [f for f in listdir(folderName) if isfile(join(folderName, f))]
        print('Arffs encontrados: ')
        print('')
        for i in arffNames: print(i)
        print('===' * 30)
        print('')
        print("Iniciando conversão..")
        print('')
        for arffs in arffNames:
            print("Carregando o arquivo " + arffs + " em memória..")
            print("")
            # Load .arff
            data = arff.load(open(folderName + arffs, 'r'))
            if norm1_int.get() == 1:
                self.normalizeFunction(data, "MinMaxScaler", folderName, arffs)
            if norm2_int.get() == 1:
                self.normalizeFunction(data, "StandardScaler", folderName, arffs)
            if norm3_int.get() == 1:
                self.normalizeFunction(data, "MaxAbsScaler", folderName, arffs)
            if norm4_int.get() == 1:
                self.normalizeFunction(data, "RobustScaler", folderName, arffs)
        print('===' * 30)
        print("Todos arquivos arffs foram convertidos.")
        print('===' * 30)

# -------------------- MAIN FUNCTION --------------------
# Instantiate the TK() Classe allowing widgets to be used
root = Tk()
# List with possible Normalization options
norm1_int = IntVar()
norm2_int = IntVar()
norm3_int = IntVar()
norm4_int = IntVar()
# Pass the root to class Application and start the Main Loop
Application(root)
root.mainloop()