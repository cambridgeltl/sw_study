# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated  01/12/2019 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import time


class SwSkipGramModel(nn.Module):
  def __init__(self, data, emb_dim, compf, use_cuda, is_sparse = True, is_v = True):
    super(SwSkipGramModel, self).__init__()
    # subword emb size
    self.emb_dim = emb_dim
    self.u_embeddings = nn.Embedding(data.mo_size + 1, emb_dim, sparse = is_sparse, padding_idx = data.mopad)
    if is_v:
      self.v_embeddings = nn.Embedding(data.vocab_size, emb_dim, sparse = is_sparse)
    self.init_emb(data.mopad, is_v)
    if 'pp' in compf:
      self.u_position_embeddings = nn.Embedding(data.pospad + 1, emb_dim, sparse = is_sparse, padding_idx = data.pospad)
      # might cause error if also initializing for mp
      self.u_position_embeddings.weight.data.zero_()
    elif 'mp' in compf:
      self.u_position_embeddings = nn.Embedding(data.pospad + 1, emb_dim, sparse = is_sparse, padding_idx = data.pospad)

    if compf.endswith('mtxatt'):
      # mtx attention
      self.da = 50
      self.r = 10
      self.WS1 = nn.Linear(emb_dim, self.da, bias = False) 
      self.WS2 = nn.Linear(self.da, self.r, bias = False)
      # another transformation layer
      #self.m2v = nn.Linear(emb_dim * self. r, self.emb_dim)
    elif compf.endswith('att'):
      # simple attention
      self.da = 50
      self.r = 1
      self.WS1 = nn.Linear(emb_dim, self.da, bias = False) 
      self.WS2 = nn.Linear(self.da, self.r, bias = False)
    # hyper parameter for attention reg term
    self.lbd = 0.15

    self.use_cuda = use_cuda
    if self.use_cuda:
      self.cuda()


  def init_emb(self, pad_idx, is_v):
    """
    Initialize embedding weight like word2vec.
    The u_embedding is a uniform distribution in [-0.5/emb_dim, 0.5/emb_dim], 
    and the elements of v_embedding are zeroes.
    """
    initrange = 0.5 / self.emb_dim
    self.u_embeddings.weight.data.uniform_(-initrange, initrange)
    self.u_embeddings.weight.data[pad_idx] = 0
    if is_v:
      self.v_embeddings.weight.data.zero_()


  def forward(self, pos_u, pos_v, neg_v, input_data, compf):
    emb_pos_u, emb_pos_v, emb_neg_v, reg_terms = self.get_word_embedding(input_data, pos_u, pos_v, neg_v, compf)

    score = torch.bmm(emb_pos_u.unsqueeze(1), emb_pos_v.unsqueeze(2))
    score = F.logsigmoid(score)

    neg_score = torch.bmm(emb_neg_v, emb_pos_u.unsqueeze(2))
    neg_score = F.logsigmoid(-1 * neg_score)
    
    return (-1 * (torch.sum(score) + torch.sum(neg_score)) + reg_terms * self.lbd) / len(pos_u)


  def get_word_embedding(self, input_data, pos_u, pos_v, neg_v, compf):
    emb_pos_u, rt_pos_u = self.get_u_embeddings(input_data, pos_u, compf)
    emb_pos_v           = self.get_v_embeddings(pos_v)
    emb_neg_v           = self.get_v_embeddings(neg_v)
    return emb_pos_u, emb_pos_v, emb_neg_v, rt_pos_u


  def get_u_embeddings(self, input_data, u_idxs, compf): 
    if compf == 'add' or compf == 'wpadd' or compf == 'wwadd' or compf == 'wwwpadd': 
      emb_u, reg_terms = self.get_u_embeddings_add(input_data, u_idxs)
    elif compf == 'ppadd' or compf == 'wwppadd':
      emb_u, reg_terms = self.get_u_embeddings_pp_add(input_data, u_idxs)
    elif compf == 'mpadd' or compf == 'wwmpadd':
      emb_u, reg_terms = self.get_u_embeddings_mp_add(input_data, u_idxs)
    else:
      sub_u, sub_mask_u, position_u = self.get_subs(u_idxs, input_data, compf) 
      sub_emb_u = self.u_embeddings(sub_u)

      if 'pp' in compf:
        position_emb_u = self.u_position_embeddings(position_u)
        sub_emb_u += position_emb_u
      elif 'mp' in compf:
        position_emb_u = self.u_position_embeddings(position_u)
        sub_emb_u *= position_emb_u

      emb_u, reg_terms = self.cal_word_embedding(sub_emb_u, sub_mask_u, compf)

    return emb_u, reg_terms


  def get_u_embeddings_add(self, input_data, word_idxs):
    sub_u = input_data.wdidx2moidx[word_idxs]
    if self.use_cuda:
      sub_u = sub_u.cuda()
    sub_emb_u = self.u_embeddings(sub_u)
    emb_u = sub_emb_u.sum(1)
    reg_terms = 0 
    return emb_u, reg_terms


  def get_u_embeddings_pp_add(self, input_data, word_idxs):
    sub_u = input_data.wdidx2moidx[word_idxs]
    position_u = input_data.wdidx2pos[word_idxs]
    if self.use_cuda:
      sub_u = sub_u.cuda()
      position_u = position_u.cuda()
    sub_emb_u = self.u_embeddings(sub_u)
    position_emb_u = self.u_position_embeddings(position_u)
    sub_emb_u += position_emb_u
    emb_u = sub_emb_u.sum(1)
    reg_terms = 0 
    return emb_u, reg_terms


  def get_u_embeddings_mp_add(self, input_data, word_idxs):
    sub_u = input_data.wdidx2moidx[word_idxs]
    position_u = input_data.wdidx2pos[word_idxs]
    if self.use_cuda:
      sub_u = sub_u.cuda()
      position_u = position_u.cuda()
    sub_emb_u = self.u_embeddings(sub_u)
    position_emb_u = self.u_position_embeddings(position_u)
    sub_emb_u *= position_emb_u
    emb_u = sub_emb_u.sum(1)
    reg_terms = 0 
    return emb_u, reg_terms


  def get_subs(self, word_idxs, input_data, compf):
    sub_idxs = input_data.wdidx2moidx[word_idxs]
    sub_masks = input_data.momask[word_idxs]
    pos_idxs = []
    if 'pp' in compf or 'mp' in compf:
      pos_idxs = input_data.wdidx2pos[word_idxs]
    if self.use_cuda:
      sub_idxs = sub_idxs.cuda()
      sub_masks = sub_masks.cuda()
      if 'pp' in compf or 'mp' in compf:
        pos_idxs = pos_idxs.cuda()
    return sub_idxs, sub_masks, pos_idxs


  def cal_word_embedding(self, sub_embs, sub_masks, setting):
    reg_terms = []
    sub_atts = self.WS1(sub_embs)
    sub_atts = F.tanh(sub_atts)
    sub_atts = self.WS2(sub_atts)
    # manual softmax with masks
    sub_atts.exp_()
    sub_atts = (sub_atts.permute(2, 0, 1) * sub_masks).permute(1, 2, 0)
    sub_atts = F.normalize(sub_atts, p = 1, dim = 1)
    word_embs = torch.bmm(sub_atts.permute(0, 2, 1), sub_embs).mean(1)
    '''
    word_embs = torch.bmm(sub_atts.permute(0, 2, 1), sub_embs).view(sub_embs.shape[0], -1)
    word_embs = self.m2v(word_embs)
    '''

    # L2 regularizer
    l2_reg = torch.pow(self.WS1.weight, 2).sum() + torch.pow(self.WS2.weight, 2).sum()
    l2_reg *= 0.15
    
    if setting.endswith('mtxatt') and self.training:
      self.ident_mtx = torch.eye(self.r).cuda() if self.use_cuda else torch.eye(self.r)
      atts_dot = torch.bmm(sub_atts.permute(0, 2, 1), sub_atts) 
      reg_terms.append((atts_dot - self.ident_mtx).norm())
      return word_embs, torch.sum(torch.stack(reg_terms)) + l2_reg

    return word_embs, l2_reg


  def get_v_embeddings(self, v_idxs):
    emb_v = self.v_embeddings(torch.LongTensor(v_idxs).cuda() if self.use_cuda else torch.LongTensor(v_idxs))
    return emb_v


  def save_embedding(self, input_data, outfile, model_file, bs, setting, word_list = None):
    """
    Save the model.
    Save all embeddings to file.
    """
    if word_list is not None:
      # only save embeddings in word list to txt file
      with open(word_list, 'r') as fin:
        words = fin.readlines()
        words = [w.strip() for w in words if w.strip()]
      word_idxs, words = zip(*[(input_data.word2idx[w], w) for w in words if w in input_data.word2idx])
      with open(outfile, 'w') as fout:
        fout.write('{} {}\n'.format(len(word_idxs), self.emb_dim))
        self.dump_emb_str(input_data, word_idxs, words, fout, setting)
    else:
      self.eval()
      word_idxs = []
      words = []
      with open(outfile, 'w') as fout:
        fout.write('{} {}\n'.format(len(input_data.idx2word), self.emb_dim))
        for word_idx, word in input_data.idx2word.items():
          word_idxs.append(word_idx)
          words.append(word)
          if len(word_idxs) < bs:
            continue 
          self.dump_emb_str(input_data, word_idxs, words, fout, setting)
          word_idxs = []
          words = []
        self.dump_emb_str(input_data, word_idxs, words, fout, setting)

      # save model
      torch.save(self.cpu(), model_file)


  def dump_emb_str(self, input_data, word_idxs, words, fout, setting):
    assert(len(word_idxs) == len(words))
    word_embs, _ = self.get_u_embeddings(input_data, word_idxs, setting)
    word_embs = word_embs.data.cpu().numpy().tolist()
    word_embs = list(zip(words, word_embs))
    word_embs = ['{} {}'.format(w[0], ' '.join(list(map(lambda x: str(x), w[1])))) for w in word_embs]
    fout.write('{}\n'.format('\n'.join(word_embs)))
