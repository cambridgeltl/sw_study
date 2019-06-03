# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  01/22/2019 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np 
import time
import os
import subprocess

import torch

from args import create_args
from model.sw2vec_model import Sw2Vec
from scripts import ios, nn_utils

import pdb

np.random.seed(1234)
torch.manual_seed(1234)


def train(m):
  t_start = time.monotonic()
  # actual learning rate
  lr = 0
  # temporary word count
  word_ct = 0
  # total word count
  word_tot_count = 0
  # word pairs for batch training
  pairs = []
  # numbers of batches
  bs_n = 0
  total_loss = .0
  # optimizer
  optimizer = nn_utils.get_optimizer(m)
  # best score
  init_test_prog = 1 * 1e-2
  test_step = 1
  best_ws = 0
  cur_ws = 0
  for i in range(m.iters):
    with open(m.data.infile, 'r') as fin:
      for line in fin:
        linevec_idx = [m.data.word2idx[w] for w in line.strip().split() if w in m.data.word2idx]
        word_ct += len(linevec_idx)
        # subsampling
        linevec_idx = [w_idx for w_idx in linevec_idx if np.random.random_sample() <= m.sub_samp_probs[w_idx]]
        if len(linevec_idx) > 1:
          pairs += m.data.get_batch_pairs(linevec_idx, m.win_size)
        if len(pairs) < m.bs:
          # not engouh training pairs
          continue

        total_loss += train_batch(m, optimizer, pairs[:m.bs], m.bs, m.neg_idxs) 
        pairs = pairs[m.bs:]
        bs_n += 1

        if word_ct > 1e4:
          word_tot_count += word_ct
          word_ct = 0
          prog = word_tot_count / (m.iters * m.data.word_ct)
          lr = nn_utils.schedule_lr(m, optimizer, prog)
          if m.validate:
            test_prog = init_test_prog * test_step
            if prog > test_prog:
              best_ws, cur_ws = getBestScore(m, best_ws)
              test_step += 1
            ios.output_states_with_scores(lr, prog, word_tot_count, m, total_loss, bs_n, t_start, best_ws, cur_ws) 
          else:
            ios.output_states(lr, prog, word_tot_count, m, total_loss, bs_n, t_start) 
          total_loss = .0
          bs_n = 0
       
  if pairs:
    while pairs:
      total_loss += train_batch(m, optimizer, pairs[:m.bs], len(pairs[:m.bs]), m.neg_idxs)
      pairs = pairs[m.bs:]
      bs_n += 1
    word_tot_count += word_ct
    prog = word_tot_count / (m.iters * m.data.word_ct)
    if m.validate:
      best_ws, cur_ws = getBestScore(m, best_ws)
      ios.output_states_with_scores(lr, prog, word_tot_count, m, total_loss, bs_n, t_start, best_ws, cur_ws) 
    else:
      ios.output_states(lr, prog, word_tot_count, m, total_loss, bs_n, t_start) 

  if not m.validate:
    print('\nOutput to file: {}\nSave model to: {}'.format(m.outfile, m.save_model))
    m.model.save_embedding(m.data, m.outfile, m.save_model, m.bs, m.compf)


def train_batch(m, optimizer, pairs, bs, neg_idxs):
  pos_u, pos_v, neg_v = gen_batch(m.data.neg_sample_probs, pairs, neg_idxs, bs, m.neg_n)

  optimizer.zero_grad()
  loss = m.model(pos_u, pos_v, neg_v, m.data, m.compf)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(filter(lambda p: p.grad is not None and p.requires_grad and not p.grad.data.is_sparse, m.model.parameters()), 1)
  optimizer.step()
  return loss.item()


def gen_batch(neg_sample_probs, pairs, neg_idxs, bs, neg_n):
  pos_u, pos_v = map(list, zip(*pairs))
  # negative sampling
  neg_v = np.random.choice(neg_idxs, (bs, neg_n), p = neg_sample_probs)
  return (pos_u, pos_v, neg_v)


def getBestScore(m, best_ws):
  m.model.save_embedding(m.data, m.outfile, m.save_model, m.bs, m.compf, m.word_list)
  cmd = ('python3 /home/yz568/Documents/code/evaluation/eval.py\
          --evaldata_path {}\
          --emb_path {}\
          --task word_similarity\
          --lang de'.format(os.path.join(os.path.dirname(m.word_list), '2.mws353.de.txt'), os.path.abspath(m.outfile)))
  out_str = subprocess.getoutput(cmd)
  #print(out_str)
  cur_ws = parse_output(out_str)
  #print(cur_ws)
  if cur_ws > best_ws:
    best_ws = cur_ws
    m.model.save_embedding(m.data, m.outfile + '.best', m.save_model + '.best', m.bs, m.compf)
  cmd = ('rm {0}.txt {0}.vocab {0}.pth'.format(os.path.abspath(m.outfile)[:-4]))
  p1 = subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
  p1.wait()
  return best_ws, cur_ws


def parse_output(output_str):
  result = output_str[output_str.find('mws353'):].strip().split('\n')[-2]
  result = float(result.strip().split()[-1])
  return result


if __name__ == '__main__':
  args = create_args()
  args.thread = 1
  sw_w2v = Sw2Vec(args)
  train(sw_w2v) 
