# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated  01/18/2019
Chararcter n-gram Pytorch Implementation
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import torch

from sw.swbase_input_data import SwBaseInputData

import pdb


class CharnBaseInputData(SwBaseInputData):
  # char n-gram threshols
  minn = 3
  maxn = 6

  def __init__(self, args):
    super(CharnBaseInputData, self).__init__(args)
    # empirical number
    self.pospad       = 512
 

  def morphvec_iter(self):
    for w in self.word2idx:
      morphvec = CharnBaseInputData.char_ngram_generator(w)
      yield w, morphvec


  @staticmethod
  def char_ngram_generator(text):
    """
    This creates the character n-grams like it is described in fasttext
    """
    z = []
    text2 = '<' + text + '>'
    for k in range(CharnBaseInputData.minn, CharnBaseInputData.maxn + 1):
      z.append([text2[i: i + k] for i in range(len(text2) - k + 1)])
    
    z = [ngram for ngrams in z for ngram in ngrams]
    if text2 in z and len(z) > 1:
      z.remove(text2)
    return z
