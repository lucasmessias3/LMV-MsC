from libs.pipeline import pipeline
from libs.utils import utils
import pandas as pd
import numpy as np
import warnings
import json
import sys

warnings.filterwarnings("ignore")

# ========== HOW TO USE ==========

# Possibles datasets: larvae, eggs, protozoan, all-15, all-16
# Possibles architectures: 0 - Inception_resnet_v2, 1 - Inception_v3, 2 - Celso
# Possibles activeSelection: 0 - RS, 1 - KN
# Example: python mainFramework-V5.py larvae 2 0

# ========== ACTIVE LEARNING TECHNIQUES ==========

# Random Sampling - Seleciona as amostras de forma aleatória
# Entropy Sampling - Seleciona as amostras com maior entropia
# Margin Sampling - Seleciona as amostras onde a diferença da classe mais provavel para a segunda mais provável é menor
# Uncertainty Sampling - Seleciona as amostras mais incertas
# RDS
# MST-BE

# ---------- MAIN ----------

# loading metadata
with open('/home/lucasmessias/MsC-Project/framework/tools/metadata.json') as json_file:
    metadata = json.load(json_file)

# loading libs
utilsObject = utils()
pipelineObject = pipeline()

# reading files
easy_X, easy_Y, hard_X, hard_Y, test_X, test_Y = utilsObject.readCSV(
    sys.argv[1],
    metadata[sys.argv[1]]["architecture"][int(sys.argv[2])],
    metadata[sys.argv[1]]["activeSelection"][int(sys.argv[3])]
)

# constraints
k_hardSamples = int(metadata[sys.argv[1]]["K"]/2)
k_easySamples = metadata[sys.argv[1]]["K"] - k_hardSamples
iteration = 5

for technique in ["Entropy Sampling", "Margin Sampling", "Uncertainty Sampling", "Average Confidence", "RDS", "MST-BE"]:

    ssmodel_accuracy, ssmodel_corrected, fullmodel_accuracy,\
        fullmodel_corrected, hard_time_to_select, easy_time_to_select,\
        ssmodel_knowClass, fullmodel_knowClass, wrong_percentage = pipelineObject.run(technique, easy_X, easy_Y, hard_X,\
                                                                        hard_Y, test_X, test_Y, k_hardSamples, k_easySamples, iteration)

#     for i in [0, 1]:
#         df = pd.DataFrame(
#             list(zip(ssmodel_accuracy[i], ssmodel_corrected[i], fullmodel_accuracy[i], fullmodel_corrected[i], hard_time_to_select[i], easy_time_to_select[i], ssmodel_knowClass[i], fullmodel_knowClass[i], wrong_percentage[i])),
#             index = np.arange(1, iteration + 1),
#             columns = ['semi-sup-accuracy', 'semi-sup-correcteds', 'sup-accuracy', 'sup-correcteds', 'hard-time-select', 'easy-time-select', 'semi-sup-knowClass', 'sup-knowClass', 'wrong-percentage'])       

#         df.to_csv("/home/lucasmessias/MsC-Project/framework/results/" +
#                     str(sys.argv[1]) + "/" +
#                     str(metadata[sys.argv[1]]["architecture"][int(sys.argv[2])]) + "-" +
#                     str(metadata[sys.argv[1]]["activeSelection"][int(sys.argv[3])]) + "-" +
#                     str(technique) + "-" +
#                     str(i + 1) + ".csv")                                                              

# fig, axs = plt.subplots(4, 5, figsize = (20, 15))
# plt.setp(axs, xticks = range(iteration))

# for ss_acc, ss_corrected, full_acc, full_corrected, method_hard, method_easy, hard_time, easy_time, ss_knowClass, full_knowClass, wrongPercentage, \
#     x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, title1, title2, title3, title4, title5 in \
#     zip(ssmodel_accuracy, ssmodel_corrected, fullmodel_accuracy, fullmodel_corrected, metadata["scenario"]["1"]["hard"], metadata["scenario"]["1"]["easy"],
#         hard_time_to_select, easy_time_to_select, ssmodel_knowClass, fullmodel_knowClass, wrong_percentage,
#         [0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3], [1, 1, 1, 1], [0, 1, 2, 3], [2, 2, 2, 2], [0, 1, 2, 3], [3, 3, 3, 3], [0, 1, 2, 3], [4, 4, 4, 4],
#         ["Accuracy", "Accuracy", "Accuracy", "Accuracy", "Accuracy"],
#         ["Corrected Samples", "Corrected Samples", "Corrected Samples", "Corrected Samples", "Corrected Samples"],
#         ["Time to Select", "Time to Select", "Time to Select", "Time to Select", "Time to Select"],
#         ["Known Classes", "Known Classes", "Known Classes", "Known Classes", "Known Classes"],
#         ["Propagated Errors %", "Propagated Errors %", "Propagated Errors %", "Propagated Errors %", "Propagated Errors %"]):

#     # Accuracy
#     axs[x1, y1].plot(range(iteration), ss_acc)
#     axs[x1, y1].plot(range(iteration), full_acc)
#     axs[x1, y1].annotate("Best Result:", xy = (0, 25))
#     axs[x1, y1].annotate("SS: " + str(max(ss_acc)) + "%", xy = (0, 18))
#     axs[x1, y1].annotate("Corrected Samples: " + str(np.array(ss_corrected[:ss_acc.index(max(ss_acc)) + 1]).sum()), xy = (0, 12), fontsize = 9)
#     axs[x1, y1].annotate("Full: " + str(max(full_acc)) + "%", xy = (0, 5))
#     axs[x1, y1].annotate("Corrected Samples: " + str(np.array(full_corrected[:full_acc.index(max(full_acc)) + 1]).sum()), xy = (0, -1), fontsize = 9)
#     axs[x1, y1].set_title(title1 + "\n( " + str(method_hard) + " - " + str(method_easy) + ")", fontsize = 11)
#     axs[x1, y1].set_xlabel('number of iterations', fontsize = 8)
#     axs[x1, y1].set_ylabel('accuracy', fontsize = 8)
#     axs[x1, y1].legend(['Semi', 'Supervised'], loc = 'lower right', fontsize = 'small')
#     axs[x1, y1].set_ylim([-5, 105])
#     axs[x1, y1].tick_params(axis = 'x', which = 'both', labelsize = 7)

#     # Corrected Samples
#     axs[x2, y2].plot(range(iteration), ss_corrected)
#     axs[x2, y2].plot(range(iteration), full_corrected)
#     axs[x2, y2].set_title(title2 + "\n( " + str(method_hard) + " - " + str(method_easy) + ")", fontsize = 11)
#     for i,j in zip(range(iteration), ss_corrected):
#         axs[x2, y2].annotate(str(j), xy = (i,j), fontsize = 8)
#     for i,j in zip(range(iteration), full_corrected):
#         axs[x2, y2].annotate(str(j), xy = (i,j), fontsize = 8)
#     axs[x2, y2].set_xlabel('number of iterations', fontsize = 8)
#     axs[x2, y2].set_ylabel('corrected samples', fontsize = 8)
#     axs[x2, y2].legend(['Semi', 'Supervised'], loc = 'upper right', fontsize = 'small')
#     axs[x2, y2].set_ylim([-0.5, int(metadata[sys.argv[1]]["K"])*2 + 0.5])
#     axs[x2, y2].tick_params(axis = 'x', which = 'both', labelsize = 7)

#     # Time to Select
#     axs[x3, y3].plot(range(iteration), hard_time)
#     axs[x3, y3].plot(range(iteration), easy_time)
#     axs[x3, y3].set_title(title3 + "\n( " + str(method_hard) + " - " + str(method_easy) + ")", fontsize = 11)
#     axs[x3, y3].set_xlabel('number of iterations', fontsize = 8)
#     axs[x3, y3].set_ylabel('time to select', fontsize = 8)
#     axs[x3, y3].legend(['Hard Samples', 'Easy Samples'], loc = 'lower right', fontsize = 'small')
#     axs[x3, y3].tick_params(axis = 'x', which = 'both', labelsize = 7)

#     # Know Classes
#     axs[x4, y4].plot(range(iteration), ss_knowClass)
#     axs[x4, y4].plot(range(iteration), full_knowClass)
#     axs[x4, y4].set_title(title4 + "\n( " + str(method_hard) + " - " + str(method_easy) + ")", fontsize = 11)
#     for i,j in zip(range(iteration), ss_knowClass):
#         axs[x4, y4].annotate(str(j), xy = (i,j), fontsize = 8)
#     for i,j in zip(range(iteration), full_knowClass):
#         axs[x4, y4].annotate(str(j), xy = (i,j), fontsize = 8)
#     axs[x4, y4].set_xlabel('number of iterations', fontsize = 8)
#     axs[x4, y4].set_ylabel('known classes', fontsize = 8)
#     axs[x4, y4].legend(['Semi', 'Supervised'], loc = 'lower right', fontsize = 'small')
#     axs[x4, y4].set_ylim([-0.5, int(metadata[sys.argv[1]]["K"])/2 + 0.5])
#     axs[x4, y4].tick_params(axis = 'x', which = 'both', labelsize = 7)

#     # Propagated Errors
#     axs[x5, y5].plot(range(iteration), wrongPercentage)
#     axs[x5, y5].set_title(title5 + "\n( " + str(method_hard) + " - " + str(method_easy) + ")", fontsize = 11)
#     for i,j in zip(range(iteration), wrongPercentage):
#         axs[x5, y5].annotate(str(j), xy = (i,j), fontsize = 6)
#     axs[x5, y5].set_xlabel('number of iterations', fontsize = 8)
#     axs[x5, y5].set_ylabel('propagated errors %', fontsize = 8)
#     axs[x5, y5].legend(['Semi'], loc = 'upper right', fontsize = 'small')
#     axs[x5, y5].set_ylim([-5, 105])
#     axs[x5, y5].tick_params(axis = 'x', which = 'both', labelsize = 8)