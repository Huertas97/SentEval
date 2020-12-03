# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:58:52 2020

This source code is adapted from SentEval dataset https://github.com/facebookresearch/SentEval.
The code allows you to evaluate Sentence Transformers models in different tasks concatenating
the sentence embeddings from different multilingual models and applying a PCA:
    
    - distiluse-base-multilingual-cased
    - xlm-r-bert-base-nli-stsb-mean-tokens
    - xlm-r-distilroberta-base-paraphrase-v1
    - LaBSE
    - distilbert-multilingual-nli-stsb-quora-ranking

If you want to use it, you have to clone this SentEval repository and download the 
data as it is explained in the examples:
    $  git clone https://github.com/Huertas97/SentEval.git
    $  cd SentEval/data/downstream
    $  bash ./get_transfer_data.bash
    
For more information about how to use this code read help information:
    $  python ensemble_dim_reduc.py --help

@author: Álvaro Huertas García
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import sys 
import pickle
import pandas as pd
from optparse import OptionParser
# Process command-line options
parser = OptionParser(add_help_option=False)

"""
Command line options
"""

# General options
parser.add_option('-d', '--data', type="string", help='Number of sentences to retrieve')
parser.add_option('-o', '--output', type="string", help='Output name')
parser.add_option('-m', '--models', type = "string", help='Languages ​​from which we extract sentences')
parser.add_option('-h', '--help', action='store_true', help='Show this help message and exit.')
(options, args) = parser.parse_args()

def print_usage():
    print("""
Usage:

    python ensembe_dim_reduc.py [options] 

Options:
    -d, --data              Path to dataframe with model names and PCAs
    -m, --models            Name models used for SentEval
    -o, --output            Output name
    -h, --help              Help information

Example. Evaluate distiluse-base-multilingual-cased and xlm-r-bert-base-nli-stsb-mean-tokens ensemble on SentEval tasks :
    python ensemble_dim_reduc.py --data df_multi_selected_99.pkl --models distiluse-base-multilingual-cased,xlm-r-bert-base-nli-stsb-mean-tokens""")
    sys.exit()

if not options.data or not options.models:
    print_usage()

if not options.output:
    options.output = options.model
    
if options.help:
    print_usage()


"""
Class to create a ensemble object with PCA for each model
"""

class ensemble_stransformer:
    
  def __init__(self, model_names, pca_list):

    # Set models and PCAs as attributes
    for i, (model_arg, pca_arg) in enumerate(zip(model_names, pca_list)):
      sentence_model = SentenceTransformer(model_arg)
      new_model_att = {"model_"+str(i): sentence_model,
                        "model_" + str(i) + "_pca": pca_arg}
      self.__dict__.update(new_model_att)

  def PCA(self, name, embedding):
    for att in dir(self):
      if name + "_pca" == att:
        emb_dim_red = getattr(self, att).transform(embedding)
        return emb_dim_red

  def encode(self, sentences):
    embeddings = []
    n_models = 0
    for att in dir(self):
      if "model_" in att and "pca" not in att:
        emb = getattr(self, att).encode(sentences)
        emb_dim_red = self.PCA(att, emb)
        embeddings.append(emb_dim_red)
        n_models += 1
    
    else:
      embeddings_concat = np.concatenate( embeddings, axis = 1)
      return embeddings_concat

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

"""
Prepare and batcher function required by SentEval
"""

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
     sentences = [' '.join(s) for s in batch]
     embeddings = params.ensemble.encode(sentences)
     return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Extract PCA fitted for the multilingual models
df_pca_data = os.join.path(PATH_TO_DATA, "PCA", options.data)
df_ensemble_pca = pd.read_pickle(df_pca_data)
models_names = df_ensemble_pca.Modelo.to_list()
models_pca = df_ensemble_pca.PCA.to_list()

# Selecting PCA from models introduced by command line
model_names = options.models.split(",")
pca_list = []
for name in model_names:
  pca = df_ensemble_pca[df_ensemble_pca["Modelo"] == name].PCA.values[0]
  pca_list.append(pca)

# Creating the models with PCA that will be used to obtain embeddings by SentEval
params_senteval["ensemble"] = ensemble_stransformer( model_names, pca_list)
                                 
# SentEval Engine
if __name__ == "__main__":

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
                      'CR', 
                      'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 
                      # 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 
                      #  'ImageCaptionRetrieval',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'
                      ]
    results = se.eval(transfer_tasks)
    
    # create result folder
    logging.debug("****** Saving results ******")
    result_path = "../result"
    if not os.path.exists(result_path):
      os.mkdir(result_path)

    save_path = os.path.join(result_path, options.output+"_result.pkl")
    pickle.dump( results, open( save_path , "wb" ) )
    logging.debug("Results saved in {}".format(save_path))
    print(results)







