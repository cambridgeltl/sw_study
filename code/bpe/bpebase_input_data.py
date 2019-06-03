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
from sw.swbase_input_data import SwBaseInputData
import sentencepiece as spm

import pdb


class BPEBaseInputData(SwBaseInputData):
  def __init__(self, args):
    super(BPEBaseInputData, self).__init__(args)
    self.splitter   = self.get_splitter()
    # bpe -> idx
    self.mo2idx     = self.get_mo_vocab()
    self.idx2mo     = {v: k for k, v in self.mo2idx.items()}
    # empirical number
    self.pospad     = 100
 

  def get_splitter(self):
    splitter = spm.SentencePieceProcessor()
    splitter.Load(self.sw_file)
    return splitter 


  def get_mo_vocab(self):
    vocab = [self.splitter.IdToPiece(i) for i in range(self.splitter.GetPieceSize())]
    # make sure every entry is a str
    vocab = [w if isinstance(w, str) else w.decode('utf-8') for w in vocab]
    sub2idx = dict(zip(vocab, list(range(len(vocab)))))
    return sub2idx


  def morphvec_iter(self):
    for w in self.word2idx:
      morphvec = self.splitter.EncodeAsPieces(w.encode('utf-8'))
      # make sure everything is str
      morphvec = [m if isinstance(m, str) else m.decode('utf-8') for m in morphvec]
      yield w, morphvec
