# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  05/15/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import time

def output_states(lr, prog, word_tot_count, m, total_loss, bs_n, t_start):
  sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Loss: %0.5f, Words/sec: %0.1f"
      % (lr,
         prog * 100,
         total_loss / bs_n,
         word_tot_count / (time.monotonic() - t_start)))
  sys.stdout.flush()


def output_states_with_scores(lr, prog, word_tot_count, m, total_loss, bs_n, t_start, best_ws, cur_ws):
  sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Loss: %0.5f, Words/sec: %0.1f, Best: %0.4f, Current: %0.4f"
    % (lr,
      prog * 100,
      total_loss / bs_n,
      word_tot_count / (time.monotonic() - t_start),
      best_ws,
      cur_ws))
  sys.stdout.flush()

