# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated  01/18/2019
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
from word.word_input_data import WordInputData
from morf.morfbase_input_data import MorfBaseInputData

import pdb


class MorfInputData(WordInputData, MorfBaseInputData):
  def __init__(self, args):
    WordInputData.__init__(self, args)
    MorfBaseInputData.__init__(self, args)

    self.gen_sw(args.compf)
    print('Morf size: {}'.format(self.mo_size)) 
    print('Max subword length: {}'.format(self.max_mo_n)) 
