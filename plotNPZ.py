#
# Created on 12/9/2020
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#
import numpy as np
import matplotlib.pyplot as plt

IM = 8
# run = 3

# modelnum = 0
# modelFolder = '/models_best_3Run_10232020'

# modelnum = 1
# modelFolder = 'models_best_3Run_12092020'

# modelnum = 3
# modelFolder = 'models_LR_0_1'

modelFolder = 'models_LR_0_01'

for IM in range(9,14):
    print('IM='+str(IM))
    for run in [1,3]:
        file = 'E:/AutomatedTracing/TraceProofreading/TraceProofreading/data/'+str(modelFolder)+'/IM='+str(IM)+'_run='+str(run)+'.npz'
        history_L = np.load(file)

        train_loss = history_L['history_train_loss']

        plt.figure()
        plt.plot(history_L['history_train_loss'])
        plt.plot(history_L['history_val_loss'])
        plt.title('Model loss IM='+str(IM)+'_run='+str(run))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        fig=plt.show()
