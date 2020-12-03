# -*- coding: utf-8 -*-

"""
Created on Fri Nov 27 16:58:52 2020

This source code is adapted from SentEval dataset https://github.com/facebookresearch/SentEval.
The code allows you to evaluate Sentence Transformers models in different tasks. 

If you want to use it, you have to clone the SentEval github and download the 
data as it is explained in the examples. 

@author: alvar
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, models
import sys 
import pickle

model_name = sys.argv[1]

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval



# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
     sentences = [' '.join(s) for s in batch]
     embeddings = params.model.encode(sentences)
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

# we define ‘bert-base-uncased’ as the word_embedding_model
word_embedding_model = models.Transformer(model_name, max_seq_length=256)
# we create a (mean) pooling layer, which then returns our sentence embedding.
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


params_senteval["model"] = model





if __name__ == "__main__":

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
      'CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 
    # 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 
                      #  'ImageCaptionRetrieval',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    pickle.dump( results, open( str(sys.argv[1])+"_result.pkl", "wb" ) )
    print(results)







