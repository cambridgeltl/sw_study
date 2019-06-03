# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated  24/01/2019
Subword model
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sms.sms_input_data import SmsInputData
from morf.morf_input_data import MorfInputData
from bpe.bpe_input_data import BPEInputData
from charn.charn_input_data import CharnInputData

from model.sw_skip_gram import SwSkipGramModel

import pdb


class Sw2Vec(object):
  def __init__(self, args):
    self.subword        = args.subword
    self.compf          = args.compf

    self.data           = self.get_data(args)

    self.outfile        = args.output 
    self.save_model     = args.save_model
    self.load_model     = args.load_model
    self.word_list      = args.word_list
    # subword embedding size
    self.emb_dim        = args.size
    self.bs             = args.batch_size
    self.win_size       = args.window
    self.iters          = args.iter
    self.lr             = args.lr
    self.neg_n          = args.negative
    # negative example indices
    self.neg_idxs       = list(range(self.data.vocab_size))
    self.sub_samp_th    = args.sample
    # subsampling distribution,  prob reserving the word
    self.sub_samp_probs = np.sqrt(self.sub_samp_th / self.data.idx2freq)  
    self.opt            = args.opt
    self.thread         = args.thread
    self.validate       = args.validate

    print('Initializing models...')
    self.init_model(args)

    info = ('Subword: {}\n'.format(self.subword) + 
        'Compf: {}\n'.format(self.compf) + 
        'Input path: {}\n'.format(args.train) + 
        'Subword model: {}\n'.format(args.sw_file) + 
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
        'Threads: {}\n'.format(self.thread) + 
        'Use CUDA: {}\n'.format(args.cuda) + 
        'Validate: {}'.format(self.validate))
    print(info)


  def get_data(self, args):
    if self.subword == 'sms':
      data = SmsInputData(args)
    elif self.subword == 'morf':
      data = MorfInputData(args)
    elif self.subword == 'bpe':
      data = BPEInputData(args)
    elif self.subword == 'charn':
      data = CharnInputData(args)
    return data


  def init_model(self, args):
    if args.cbow == 0:
      if self.lr == -1.0:
        self.lr = 0.05
      if self.load_model is not None:
        print('Loading model from: {}...'.format(self.load_model))
        self.model = torch.load(self.load_model) 
        self.model.train()
      else: 
        self.model = SwSkipGramModel(self.data, self.emb_dim, self.compf, args.cuda)
