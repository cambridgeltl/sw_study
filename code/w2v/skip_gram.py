# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  15/05/2018 
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


class SkipGramModel(nn.Module):
  def __init__(self, vocab_size, emb_dim):
    super(SkipGramModel, self).__init__()
    self.emb_dim = emb_dim
    self.u_embeddings = nn.Embedding(vocab_size, emb_dim, sparse = True)
    self.v_embeddings = nn.Embedding(vocab_size, emb_dim, sparse = True)
    self.init_emb()


  def init_emb(self):
    """
    Initialize embedding weight like word2vec.
    The u_embedding is a uniform distribution in [-0.5/emb_dim, 0.5/emb_dim], 
    and the elements of v_embedding are zeroes.
    """
    initrange = 0.5 / self.emb_dim
    self.u_embeddings.weight.data.uniform_(-initrange, initrange)
    self.v_embeddings.weight.data.zero_()


  def forward(self, pos_u, pos_v, neg_v, m):
    emb_pos_u, emb_pos_v, emb_neg_v = self.get_word_embedding(m.data, pos_u, pos_v, neg_v, m.use_cuda)
    
    loss = self.calc_loss(emb_pos_u, emb_pos_v, emb_neg_v)
    return loss


  def get_word_embedding(self, input_data, pos_u, pos_v, neg_v, use_cuda):
    emb_pos_u = self.get_u_embeddings(input_data, pos_u, use_cuda)
    emb_pos_v = self.get_v_embeddings(input_data, pos_v, use_cuda)
    emb_neg_v = self.get_v_embeddings(input_data, neg_v, use_cuda)

    return emb_pos_u, emb_pos_v, emb_neg_v


  def get_u_embeddings(self, input_data, u_idxs, use_cuda):
    emb_u = self.u_embeddings(torch.LongTensor(u_idxs).cuda() if use_cuda else torch.LongTensor(u_idxs))
    return emb_u


  def get_v_embeddings(self, input_data, v_idxs, use_cuda):
    emb_v = self.v_embeddings(torch.LongTensor(v_idxs).cuda() if use_cuda else torch.LongTensor(v_idxs))
    return emb_v


  def calc_loss(self, emb_pos_u, emb_pos_v, emb_neg_v):
    score = torch.bmm(emb_pos_u.unsqueeze(1), emb_pos_v.unsqueeze(2))
    score = F.logsigmoid(score)

    neg_score = torch.bmm(emb_neg_v, emb_pos_u.unsqueeze(2))
    neg_score = F.logsigmoid(-1 * neg_score)

    loss = -1 * (torch.sum(score) + torch.sum(neg_score)) / emb_pos_u.size()[0]
    return loss


  def save_embedding(self, m):
    """
    Save all embeddings to file.
    Save the model.
    """
    self.eval()
    torch.save(self, m.save_model)

    word_idxs = []
    words = []
    with open(m.outfile, 'w') as fout:
      fout.write('{} {}\n'.format(len(m.data.idx2word), self.emb_dim))
      for word_idx, word in m.data.idx2word.items():
        word_idxs.append(word_idx)
        words.append(word)
        if len(word_idxs) < m.bs:
          continue 
        self.dump_emb_str(m.data, word_idxs, words, fout, m.use_cuda)
        word_idxs = []
        words = []
      self.dump_emb_str(m.data, word_idxs, words, fout, m.use_cuda)


  def dump_emb_str(self, input_data, word_idxs, words, fout, use_cuda):
    assert(len(word_idxs) == len(words))
    word_embs = self.get_u_embeddings(input_data, word_idxs, use_cuda)
    word_embs = word_embs.data.cpu().numpy().tolist()
    word_embs = list(zip(words, word_embs))
    word_embs = ['{} {}'.format(w[0], ' '.join(list(map(lambda x: str(x), w[1])))) for w in word_embs]
    fout.write('{}\n'.format('\n'.join(word_embs)))
