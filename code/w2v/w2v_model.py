# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  22/05/2018 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import time
import sys

from input_data import InputData
from skip_gram import SkipGramModel

import pdb


class Word2Vec(object):
  def __init__(self, args):
    self.thread         = args.thread if 'thread' in args else 1
    # data class
    self.data           = self.get_data(args)

    self.outfile        = args.output                                       
    self.save_model     = args.save_model
    self.load_model     = args.load_model

    self.emb_dim        = args.size                                         
    self.bs             = args.batch_size
    self.win_size       = args.window
    self.iters          = args.iter
    self.lr             = args.lr
    self.gclip          = args.gclip
    self.neg_n          = args.negative
    # negative example indices
    self.neg_idxs       = list(range(self.data.vocab_size))
    self.sub_samp_th    = args.sample
    #subsampling,  prob reserving the word
    self.sub_samp_probs = np.sqrt(self.sub_samp_th / self.data.idx2freq)      
    self.opt            = args.opt
    self.use_cuda       = args.cuda

    self.print_model_info()


  def get_data(self, args):
    return InputData(args.train, args.min_count, self.thread)


  def init_model(self, args):
    print('Initializing models...')
    if args.cbow == 0:
      if self.load_model is not None:
        print('Loading model from: {}...'.format(self.load_model)) 
        self.model = torch.load(self.load_model) 
        self.model.train()
      else: 
        self.model = self.get_model()
    if self.use_cuda:
      self.model.cuda()
  

  def get_model(self):
    return SkipGramModel(self.data.vocab_size, self.emb_dim)


  def print_model_info(self):
    info = ( 
        '-' * 30 + 'Model Info' + '-' * 30 +'\n' + 
        'Input path: {}\n'.format(self.data.infile) + 
        'Output path: {}\n'.format(self.outfile) + 
        'Save model to: {}\n'.format(self.save_model) + 
        'Embedding dim: {}\n'.format(self.emb_dim) + 
        'Batch size: {}\n'.format(self.bs) + 
        'Window size: {}\n'.format(self.win_size) + 
        'Negative examples: {}\n'.format(self.neg_n) + 
        'Subsampling rate: {}\n'.format(self.sub_samp_th) + 
        'Iterations: {}\n'.format(self.iters) + 
        'Optimizer: {}\n'.format(self.opt) + 
        'Learning rate: {}\n'.format(self.lr) + 
        'Gradient clipping: {}\n'.format(self.gclip) + 
        'Use CUDA: {}\n'.format(self.use_cuda) + 
        'Thread: {}\n'.format(self.thread) + 
        '-' * 70 
        )
    print(info)
