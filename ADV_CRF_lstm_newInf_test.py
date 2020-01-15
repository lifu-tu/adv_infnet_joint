import sys
import warnings
from utils import getWordmap
from utils import getData
from utils import getTagger
import random
import numpy as np
random.seed(1)
np.random.seed(1)
import argparse
parser = argparse.ArgumentParser()



parser.add_argument("-eta", help="Learning rate", type=float, default=0.05)
parser.add_argument("-l2", help="L2 regularization hyperparater", type=float, default=0.00001)
parser.add_argument("-l3", help="regularization hyperparater", type=float, default=5)
parser.add_argument("-Lambda", help="InfNet regularization hyperparater", type=int, default=10)
parser.add_argument("-batchsize", help="Size of batch when training", type=int, default=10)
parser.add_argument("-emb", help="0:fix embedding, 1:update embedding", type=int, default=0)
parser.add_argument("-seed", help="random seed", type=int, default=1)
parser.add_argument("-CostAugmentInfType", help="0:share feature network, 1:include label input 2:separate networks ", type=int, default=0)
parser.add_argument("-dropout", help="dropout", type=int, default=0)
parser.add_argument("-regu_type", help="0:local cross entropy, 1:entropy 2:Regularization Toward Pretrained Inference Network", type=int, default=0)
parser.add_argument("-annealing", help="0:fixed weght, 1:weight annealing", type=int, default=0)
parser.add_argument("-margin_type", help="different traing method  0:margin rescaling, 1:contrastive, 2:perceptron, 3: slack rescaling", type=int, default=0)
params = parser.parse_args()

params.dataf = '../pos_data/oct27.traindev.proc.cnn'
params.dev = '../pos_data/oct27.test.proc.cnn'
params.test = '../pos_data/daily547.proc.cnn'
params.hidden = 100
params.embedsize = 100


(words, We) = getWordmap('wordvects.tw100w5-m40-it2')
We = np.asarray(We).astype('float32')
tagger = getTagger('../pos_data/tagger')
params.outfile = 'inf_g_norm_'+str(params.batchsize)+'g_gradientnorm_CostAugmentInfType_'+ str(params.CostAugmentInfType) + '_Lambda_'+str(params.Lambda)  + '_LearningRate_'+str(params.eta)+ '_' + str(params.l2)+ '_'  + str(params.l3)  + '_emb_'+ str(params.emb) + '_regu_type_'+ str(params.regu_type) + '_margin_type_'+ str(params.margin_type) + '_seed_'+ str(params.seed)
	
traindata = getData(params.dataf, words, tagger)
trainx0, trainy0 = traindata
devdata = getData(params.dev, words, tagger)
devx0, devy0 = devdata
testdata = getData(params.test, words, tagger)
testx0, testy0 = testdata	



if params.CostAugmentInfType==0:
      from ADV_CRF_lstm_newInf_shared  import GAN_CRF_model
elif (params.CostAugmentInfType==1):
      from ADV_CRF_lstm_newInf_stacked_bilinear import GAN_CRF_model
elif (params.CostAugmentInfType==2):
      from ADV_CRF_lstm_newInf_separate import GAN_CRF_model
elif (params.CostAugmentInfType==3):
      from ADV_CRF_lstm_newInf_stacked_BLSTM import GAN_CRF_model
#elif (params.CostAugmentInfType==4):
#      from ADV_CRF_lstm_newInf_perp import GAN_CRF_model
elif (params.CostAugmentInfType==4):
      from ADV_CRF_lstm_newInf_margin import GAN_CRF_model

tm = GAN_CRF_model(We, params)
tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)
