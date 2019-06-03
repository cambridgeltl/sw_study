# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated  01/21/2019
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
from sw.swbase_input_data import SwBaseInputData


class SmsBaseInputData(SwBaseInputData):
  def __init__(self, args):
    super(SmsBaseInputData, self).__init__(args)
    # dict including 0(prefix), 1(stem), 2(suffix), 3(other), 4(word), 5(pad)
    self.pospad       = 5       

  
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


  def update_morph_idx(self, compf, w, morphvec):
    morphvec = SmsBaseInputData.update_morphvec(compf, morphvec)
    super(SmsBaseInputData, self).update_morph_idx(compf, w, morphvec)


  @staticmethod
  def update_morphvec(compf, morphvec):
    if compf.startswith('wp'):
      # assign word:morphtag into morph embeddings
      for i, mo in enumerate(morphvec):
        mo = mo.strip()
        col_idx = mo.rfind(':')
        if 'ROOT' in mo:
          morphvec[i] = mo[:col_idx] + ':ROOT'
        elif 'PREFIX' in mo:
          morphvec[i] = mo[:col_idx] + ':PREFIX'
        elif 'SUFFIX' in mo:
          morphvec[i] = mo[:col_idx] + ':SUFFIX'
        elif 'WORD' in mo:
          morphvec[i] = mo[:col_idx] + ':WORD'
        else:
          morphvec[i] = mo[:col_idx] + ':OTHER'
    else :
      # not wp, embed normal subword
      morphvec = [m.strip()[:m.strip().rfind(':')] for m in morphvec]
    return morphvec


  def update_pos_idx(self, compf, w, morphvec):
    aff_idx = SmsBaseInputData.get_pos_idx(compf, morphvec)
    if compf.startswith('pp') or compf.startswith('mp'):
      self.wdidx2pos[self.word2idx[w]] = aff_idx


  @staticmethod
  def get_pos_idx(compf, morphvec):
    aff_idx = []
    if compf.startswith('pp') or compf.startswith('mp'):
      # extract affix idxs
      for mo in morphvec:
        if 'ROOT' in mo:
          aff = 1
        elif 'PREFIX' in mo:
          aff = 0
        elif 'SUFFIX' in mo:
          aff = 2
        elif 'WORD' in mo:
          aff = 4
        else:
          aff = 3
        aff_idx.append(aff)
    return aff_idx
