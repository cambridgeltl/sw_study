# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  31/05/2018 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import time
import sys
from multiprocessing import set_start_method

import torch
import torch.multiprocessing as mp

sys.path.append('../scripts')
import ios, nn_utils
from args import create_w2v_parser, add_mul_args, parse_args
from w2v_model import Word2Vec
from word2vec import train_batch, get_pairs, get_progress

np.random.seed(1234)
torch.manual_seed(1234)


def train(m):
  # total count
  word_tot_count = mp.Value('L', 0)
  real_word_ct = mp.Value('L', 0)
  processes = []
  for p_id in range(m.thread):
    p = mp.Process(target = train_process, args = (p_id, real_word_ct, word_tot_count, m))
    processes.append(p)
    p.start()
  for p in processes:
    p.join()
  print('\nOutput to file: {}\nSave model to: {}'.format(m.outfile, m.save_model)) 
  m.model.save_embedding(m)


def train_process(p_id, real_word_ct, word_tot_count, m):  
  # get texts according to p_id
  text = get_text_from_pid(p_id, m)
  # get real word count
  with real_word_ct.get_lock():
    real_word_ct.value += sum([len([w for w in line.strip().split() if w in m.data.word2idx]) for line in text]) 
  
  t_start = time.monotonic()
  lr = 0
  word_ct = 0
  prev_word_ct = 0
  # positive pairs for batch training
  pairs = []
  # numbers of batches
  bs_n = 0
  # total loss for a checkpoint
  total_loss = .0
  # optimizer
  optimizer = nn_utils.get_optimizer(m)
  for i in range(m.iters):
    for line in text:
      word_ct = get_pairs(m, pairs, line, word_ct)
      if len(pairs) < m.bs:
        # not engouh training pairs
        continue

      total_loss += train_batch(m, optimizer, pairs[:m.bs], m.bs) 
      pairs = pairs[m.bs:]
      bs_n += 1
      with word_tot_count.get_lock():
        word_tot_count.value += word_ct
      word_ct = 0

      if word_tot_count.value - prev_word_ct > 1e5:
        prog = word_tot_count.value / (m.iters * real_word_ct.value)
        lr = nn_utils.schedule_lr(m, optimizer, prog)
        ios.output_states(lr, prog, word_tot_count.value, m, total_loss, bs_n, t_start) 
        total_loss = .0
        bs_n = 0
        prev_word_ct = word_tot_count.value
               
  if pairs:
    while pairs:
      total_loss += train_batch(m, optimizer, pairs[:m.bs], len(pairs[:m.bs]))
      pairs = pairs[m.bs:]
      bs_n += 1
    with word_tot_count.get_lock():
      word_tot_count.value += word_ct
    prog = word_tot_count.value / (m.iters * real_word_ct.value)
    ios.output_states(lr, prog, word_tot_count.value, m, total_loss, bs_n, t_start) 


def get_text_from_pid(p_id, m):
  start_pos = m.data.start_pos[p_id]
  end_pos = m.data.end_pos[p_id]
  with open(m.data.infile) as fin:
    if p_id == 0:
      fin.seek(0)
    else:
      fin.seek(start_pos + 1)
    if end_pos is None:
      text = fin.read().strip().split('\n')
    else:
      nbytes = end_pos - start_pos + 1
      text = fin.read(nbytes).strip().split('\n')
  return text


if __name__ == '__main__':
  set_start_method('spawn')
  
  parser = create_w2v_parser()
  parser = add_mul_args(parser)
  args = parse_args(parser)

  w2v = Word2Vec(args)
  w2v.init_model(args)

  train(w2v) 
