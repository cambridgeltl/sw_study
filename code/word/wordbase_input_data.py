#-*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated  01/21/2019
"""

#************************************************************
# Imported Libraries
#************************************************************
from abc import ABC


class WordBaseInputData(ABC):
  def __init__(self, args): 
    # word <-> idx 
    self.word2idx   = {}
    self.idx2word   = {}
    self.vocab_size = 0 
