# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated  01/22/2019
Chararcter n-gram Pytorch Implementation
"""

#************************************************************
# Imported Libraries
#************************************************************
from word.word_input_data import WordInputData
from charn.charnbase_input_data import CharnBaseInputData

import pdb


class CharnInputData(WordInputData, CharnBaseInputData):
  def __init__(self, args):
    WordInputData.__init__(self, args)
    CharnBaseInputData.__init__(self, args)

    self.gen_sw(args.compf)
    print('Char N-gram size: {}'.format(self.mo_size))
    print('Max subword length: {}'.format(self.max_mo_n)) 
