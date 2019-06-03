# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  08/12/2018 
Run Morfessor
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse
import pdb
import os
import sys
import morfessor
import pickle


def create_args():
  parser = argparse.ArgumentParser(description = 'Supervised Morhpological Segmentation subword')
  # Experiment Settings
  parser.add_argument("--train", action = 'store_true',
      help = 'training mode or test mode')
  parser.add_argument('--train_data', type = str, default = '',
      help = 'training file path')
  # data to be segmented, the data is ONE WORD PER LINE!
  parser.add_argument('--test_data', type = str, default = '',
      help = 'the data to be parsed')
  parser.add_argument('--output', type = str, default = '',
      help = 'output file')
  parser.add_argument("--save_model", type = str, default = 'model.bin',
      help = 'saving model path')
  parser.add_argument("--load_model", type = str, default = 'model.bin',
      help = 'loading model path')
  parser.add_argument("--min_count", type = int, default = 5,
      help = 'minimum count')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = create_args()
  if args.train == True:
    # training
    io = morfessor.MorfessorIO()
    train_data = list(io.read_corpus_file(args.train_data))
    model_tokens = morfessor.BaselineModel()
    model_tokens.load_data(train_data, freqthreshold = args.min_count)
    model_tokens.train_batch()
    with open(args.save_model, 'wb') as fout:
      pickle.dump(model_tokens, fout)
  else:
    # inference
    with open(args.load_model, 'rb') as fin:
      model_tokens = pickle.load(fin)
    # test file and training file are different
    with open(args.output, 'w') as fout, open(args.test_data, 'r') as fin:
      for line in fin:
        line = line.strip()
        morph_list, score = model_tokens.viterbi_segment(line)
        morphs = ' '.join(morph_list)
        fout.write('{} {}\n'.format(line, morphs))
