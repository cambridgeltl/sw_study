# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated  01/22/2019
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
from sw.swbase_input_data import SwBaseInputData

import pdb


class MorfBaseInputData(SwBaseInputData):
  def __init__(self, args):
    super(MorfBaseInputData, self).__init__(args)
    # empirical number
    self.pospad       = 100


  def morphvec_iter(self):
    with open(self.sw_file, 'r') as fin:
      for line in fin:
        linevec = line.strip().split()
        w = linevec[0]
        # has to be in the vocab file
        if w not in self.word2idx:
          continue
        morphvec = linevec[1:]
        yield w, morphvec
