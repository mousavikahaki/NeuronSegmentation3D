#
# Created on 9/17/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#
import numpy as np
from scipy import sparse
import scipy.io as sio
import matlab.engine
import matplotlib.pyplot as plt

class IM3D:
    # init method or constructor, to pass automatically
    def __init__(self, IM, IM_Max = None):
        self.IM = IM
        self.IM_Max = IM_Max

    def Z_Projection(self):
        IM_Max = np.zeros((len(self), len(self[0])))
        for i in range(len(self)):
            for j in range(len(self[0])):
                IM_Max[i, j] = np.amax(self[i, j, :])
        return IM_Max

    def plt(self):
        IM_Max = self
        plt.figure()
        plt.imshow(IM_Max, cmap='gray')
        plt.show()
        plt.draw()
        return plt


class Trace:
    def __init__(self, AM, r, IM):
        self.AM = AM
        self.r = r
        self.IM = IM
        self.var = {}

    def plt(self):
        print(self.AM.shape)
        self.var['AM_BP'] = np.asarray(self.AM)
        self.var['r'] = self.r
        self.var['IM'] = self.IM
        sio.savemat('temp.mat', self.var)
        eng = matlab.engine.start_matlab()
        eng.evalc("s = load('temp.mat');figure;imshow(max(s.IM,[],3));hold on;PlotAM_1(s.AM_BP{1}, s.r)")
        return eng

    def GetBranch(self):
        AM_BP = np.zeros((self.AM.shape))
        BP = []
        AM_G_A = self.AM.toarray()
        for i in range(self.AM.shape[1]):
            maxvalue = np.count_nonzero(AM_G_A[i, :])
            if maxvalue > 2:
                BP.append(i)
                AM_BP[i, :] = AM_G_A[i, :]
                AM_BP[:, i] = AM_G_A[:, i]
        return sparse.csr_matrix(AM_BP)

    def removeBranches(self):
        AM_rem_br = self.AM.toarray()
        for i in range(len(AM_rem_br)):
            maxvalue = np.count_nonzero(AM_rem_br[i,:])
            if maxvalue > 2:
                # BP.append(i)
                AM_rem_br[i, :] = 0
                AM_rem_br[:, i] = 0
        AM_rem_br = np.asarray(AM_rem_br)
        return sparse.csr_matrix(AM_rem_br)

    @classmethod
    def loadTrace(self,path):
        G = sio.loadmat(path)
        IM = G['IM']
        AM = G['AM']
        r = G['r']
        R = G['R']
        return IM,AM,r,R


class cl_scenario:
    k = 1 # instant variable
    emptyElementValue = 0.5#np.inf
    def __init__(self, maxNumPoints, scenariosShape,scenario,cluster_r):
        self.maxNumPoints = maxNumPoints
        self.scenariosShape = scenariosShape
        self.scenario = scenario
        self.cluster_r = cluster_r

    def get_endpoint_scenario_features(self):
        inputSize = int((self.maxNumPoints * (self.maxNumPoints+6)))
        features_arr = np.ones(inputSize)
        features_arr[:] = self.emptyElementValue
        features_arr[0:len(self.scenario.flatten())] = self.scenario.flatten()
        return features_arr

    def get_endpoint_features(self):
        inputSize = int((self.maxNumPoints * 6))
        features_arr = np.ones(inputSize)
        features_arr[:] = self.emptyElementValue
        features_arr[0:len(self.scenario.flatten())] = self.scenario.flatten()
        return features_arr

    # Regular method # automatically take the instance (i.e. S1,S2) as the first input
    def getUpperArr(self):
        inputSize = int((self.maxNumPoints * (self.maxNumPoints - 1))/2)
        upperTriangle = self.scenario[np.triu_indices(self.scenariosShape, k=1)]
        # print(upperTriangle.shape)
        scenario_arr = np.ones(inputSize)
        scenario_arr[:] = self.emptyElementValue
        scenario_arr[0:len(upperTriangle)] = upperTriangle
        return scenario_arr

    def getWholeArr(self):
        inputSize = int(self.maxNumPoints * self.maxNumPoints)
        scenario_arr = np.ones(inputSize)
        scenario_arr[:] = self.emptyElementValue
        scenario_arr[0:len(self.scenario.flatten())] = self.scenario.flatten()
        return scenario_arr

    def getArrwith_r1(self):
        inputSize = int((self.maxNumPoints * self.maxNumPoints)+(self.maxNumPoints * 3))
        scenario_arr = np.ones(inputSize)
        scenario_arr[:] = self.emptyElementValue
        scenario_arr[0:len(self.scenario.flatten())] = self.scenario.flatten()
        return scenario_arr

    def getArrwith_r(self):
        inputSize = int((self.maxNumPoints * self.maxNumPoints))
        scenario_arr = np.ones(inputSize)
        scenario_arr[:] = self.emptyElementValue
        scenario_arr[0:len(self.scenario.flatten())] = self.scenario.flatten()

        inputSize1 = int((self.maxNumPoints * 3))
        scenario_arr1 = np.ones(inputSize1)
        scenario_arr1[:] = self.emptyElementValue
        scenario_arr1[0:len(self.cluster_r.flatten())] = self.cluster_r.flatten()
        scenario_arrFinal = np.concatenate((scenario_arr,scenario_arr1), axis=0)

        return scenario_arrFinal

    # # class method # To work with CLASS information # NOT automatically take the instance (i.e. S1,S2) as the first input
    # @classmethod    # <----- Decorator
    # def classmethod(cls, amount): # clc is class variable
    #     cls.raise_amt = amount
    #
    # @staticmethod
    # def staticmethod():
    #     return 'static method called'
