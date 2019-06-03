## -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated  09/18/2018 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import time

import torch
import torch.multiprocessing as mp

from args import create_args
from sw2vec import train_batch, getBestScore
from model.sw2vec_model import Sw2Vec
from scripts import ios, nn_utils


import pdb

np.random.seed(1234)
torch.manual_seed(1234)


def train(m):
  # total count
  word_tot_count = mp.Value('L', 0)
  real_word_ct = mp.Value('L', 0)
  init_test_prog = mp.Value('d', 1 * 1e-2)
  test_step = mp.Value('i', 1)
  # best score in evaluation
  best_ws = mp.Value('d', .0)
  # current scores for validation
  cur_ws = mp.Value('d', .0)
  # lock for evaluation
  lock = mp.Lock()
  processes = []
  for p_id in range(m.thread):
    p = mp.Process(target = train_process, args = (p_id, lock, real_word_ct, word_tot_count, m, init_test_prog, test_step, best_ws, cur_ws))
    processes.append(p)
    p.start()
  for p in processes:
    p.join()
  if not m.validate:
    print('\nOutput to file: {}\nSave model to: {}'.format(m.outfile, m.save_model))
    m.model.save_embedding(m.data, m.outfile, m.save_model, m.bs, m.compf)


def train_process(p_id, lock, real_word_ct, word_tot_count, m, init_test_prog, test_step, best_ws, cur_ws):
  # get texts according to p_id
  text = get_text_from_pid(p_id, m)
  # get real word count
  with real_word_ct.get_lock():
    real_word_ct.value += sum([len([w for w in line.strip().split() if w in m.data.word2idx]) for line in text])

  t_start = time.monotonic()
  # actual learning rate
  lr = 0
  word_ct = 0
  prev_word_ct = 0
  # positive pairs for batch training
  pairs = []
  # numbers of batches
  bs_n = 0
  # total loss
  total_loss = .0
  # optimizer
  optimizer = nn_utils.get_optimizer(m)

  for i in range(m.iters):
    for line in text:
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
      with word_tot_count.get_lock():
        word_tot_count.value += word_ct
      word_ct = 0

      if word_tot_count.value - prev_word_ct > 1e5: 
        prog = word_tot_count.value / (m.iters * real_word_ct.value)
        lr = nn_utils.schedule_lr(m, optimizer, prog)
        if m.validate:
          with test_step.get_lock(), best_ws.get_lock():
            if prog > init_test_prog.value * test_step.value:
              lock.acquire(True)
              try:
                best_ws.value, cur_ws.value = getBestScore(m, best_ws.value)
                test_step.value += 1
              finally:
                lock.release()
          ios.output_states_with_scores(lr, prog, word_tot_count.value, m, total_loss, bs_n, t_start, best_ws.value, cur_ws.value) 
        else:
          ios.output_states(lr, prog, word_tot_count.value, m, total_loss, bs_n, t_start) 
        total_loss = .0
        bs_n = 0
        prev_word_ct = word_tot_count.value

  if pairs:
    while pairs:
      total_loss += train_batch(m, optimizer, pairs[:m.bs], len(pairs[:m.bs]), m.neg_idxs)
      pairs = pairs[m.bs:]
      bs_n += 1
    with word_tot_count.get_lock():
      word_tot_count.value += word_ct
    prog = word_tot_count.value / (m.iters * real_word_ct.value)
    if m.validate:
      with best_ws.get_lock():
        lock.acquire(True)
        try:
          best_ws.value, cur_ws.value = getBestScore(m, best_ws.value)
        finally:
          lock.release()
      ios.output_states_with_scores(lr, prog, word_tot_count.value, m, total_loss, bs_n, t_start, best_ws.value, cur_ws.value) 
    else:
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
  mp.set_start_method('spawn')
  args = create_args()
  sw_w2v = Sw2Vec(args)
  train(sw_w2v) 
