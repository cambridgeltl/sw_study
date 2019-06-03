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
from abc import ABC, abstractmethod
import torch

class SwBaseInputData(ABC):
  def __init__(self, args):
    # segmentation methods
    self.subword      = args.subword
    # Subword part
    self.sw_file      = args.sw_file
     
    # subwords <-> idx
    self.mo2idx       = {}
    self.idx2mo       = {}
    # subword size
    self.mo_size      = None 
    # subword padding
    self.mopad        = None
    
    # pos padding
    self.pospad       = None
    # word_id -> [subword seq id]
    self.wdidx2moidx  = None
    # word_id -> [position seq]
    self.wdidx2pos    = None
    # [subword seq mask]
    self.momask       = None
    # max# of subword
    self.max_mo_n     = 0


  def gen_sw(self, compf):
    self.wdidx2moidx = {}
    self.wdidx2pos = {}
    self.momask = []
    ww, compf = SwBaseInputData.if_add_word(compf)

    for w, morphvec in self.morphvec_iter():
      morphvec = SwBaseInputData.get_morphvec(w, morphvec, self.subword, compf, ww)
      mo_n = len(morphvec)
      if mo_n > self.max_mo_n:
        self.max_mo_n = mo_n
      # update morph_idx and pos_idx table
      self.update_morph_idx(compf, w, morphvec)
      self.update_pos_idx(compf, w, morphvec)
 
    self.wdidx2moidx = [tp[1] for tp in sorted(self.wdidx2moidx.items(), key = lambda x: x[0])]
    if compf.startswith('pp') or compf.startswith('mp'):
      self.wdidx2pos = [tp[1] for tp in sorted(self.wdidx2pos.items(), key = lambda x: x[0])]
    self.mo_size = len(self.mo2idx)
    self.mopad = self.mo_size

    # padding
    self.pad_morph(compf)


  @staticmethod
  def if_add_word(compf):
    # w/wo word embeddings
    ww = False
    if compf.startswith('ww'):
      # with word embeddings
      compf = compf[2:]
      ww = True
    return ww, compf


  @abstractmethod
  def morphvec_iter(self):
    pass


  @staticmethod
  def get_morphvec(w, morphvec, subword, compf, ww):
    # add < and > to the beginning and end
    if subword == 'sms':
      first_idx = morphvec[0].rfind(':') 
      morphvec[0] = '<' + morphvec[0][:first_idx] + morphvec[0][first_idx:]
      last_idx = morphvec[-1].rfind(':')
      morphvec[-1] = morphvec[-1][:last_idx] + '>' + morphvec[-1][last_idx:]
    elif subword == 'morf' or subword == 'bpe':
      morphvec[0] = '<' + morphvec[0]
      morphvec[-1] = morphvec[-1] + '>'
    if ww:
      if subword == 'sms':
        # add word anyway for wp, pp, mp, for others unless there are subwords
        if compf.startswith('wp') or \
           compf.startswith('pp') or \
           compf.startswith('mp'):
          morphvec.append('{}:WORD'.format('<' + w + '>'))
        elif len(morphvec) > 1:
          morphvec.append('{}:WORD'.format('<' + w + '>'))
      elif len(morphvec) > 1:
        morphvec.append('<' + w + '>')
    return morphvec


  def update_morph_idx(self, compf, w, morphvec):
    for m in morphvec:
      if m not in self.mo2idx:
        cur_idx = len(self.mo2idx)
        self.mo2idx[m] = cur_idx
        self.idx2mo[cur_idx] = m
    morph_idx = [self.mo2idx[m] for m in morphvec]
    self.wdidx2moidx[self.word2idx[w]] = morph_idx


  def update_pos_idx(self, compf, w, morphvec):
    pos_idx = SwBaseInputData.get_pos_idx(compf, morphvec)
    if compf.startswith('pp') or compf.startswith('mp'):
      self.wdidx2pos[self.word2idx[w]] = pos_idx


  @staticmethod
  def get_pos_idx(compf, morphvec):
    pos_idx = []
    if compf.startswith('pp') or compf.startswith('mp'):
      pos_idx = list(range(len(morphvec)))
    return pos_idx


  def pad_morph(self, compf):
    for i, morph_idx in enumerate(self.wdidx2moidx): 
      mo_n = len(morph_idx)
      # padding
      self.wdidx2moidx[i] = morph_idx + [self.mopad] * (self.max_mo_n - mo_n)
      self.momask.append([1] * mo_n + [0] * (self.max_mo_n - mo_n))
      # position padding
      if compf.startswith('pp') or compf.startswith('mp'):
        self.wdidx2pos[i] = self.wdidx2pos[i] + [self.pospad] * (self.max_mo_n - mo_n)
    self.wdidx2moidx = torch.LongTensor(self.wdidx2moidx)
    self.momask = torch.Tensor(self.momask)
    if compf.startswith('pp') or compf.startswith('mp'):
      self.wdidx2pos = torch.LongTensor(self.wdidx2pos)


if __name__ == '__main__':
  import sys
  import os
  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

  import torch 
  from args import create_args
  from sms.sms_input_data import SmsInputData
  from morf.morf_input_data import MorfInputData
  from bpe.bpe_input_data import BPEInputData
  from charn.charn_input_data import CharnInputData

  args = create_args()
  if args.subword == 'sms':
    data = SmsInputData(args)
  elif args.subword == 'morf':
    data = MorfInputData(args)
  elif args.subword == 'bpe':
    data = BPEInputData(args)
  elif args.subword == 'charn':
    data = CharnInputData(args)
  state_dict = {'mo2idx': data.mo2idx,
                'word2idx': data.word2idx}
  torch.save(state_dict, args.save_model)
