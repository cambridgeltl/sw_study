# -*- coding: UTF-8 -*-
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
import argparse
import torch


def create_args():
  parser = argparse.ArgumentParser(description = 'Subword embedding models')
  # subword segmentations
  parser.add_argument("--subword", type = str, default = 'bpe', 
      choices = ['sms', 'morf', 'bpe', 'charn'],
      help = "subword segmentation methods")

  # composition functions
  parser.add_argument("--compf", type = str, default = 'add', 
      choices = ['add', 'wpadd', 'ppadd', 'mpadd', 
                 'wwadd', 'wwwpadd', 'wwppadd', 'wwmpadd',
                 'att', 'wpatt', 'ppatt', 'mpatt', 
                 'wwatt', 'wwwpatt', 'wwppatt', 'wwmpatt',
                 'mtxatt', 'wpmtxatt', 'ppmtxatt', 'mpmtxatt', 
                 'wwmtxatt', 'wwwpmtxatt', 'wwppmtxatt', 'wwmpmtxatt'],
      help = "composition function")

  # The following arguments are mandatory
  parser.add_argument("--train", type = str, required = True, 
      help = "training file")
  parser.add_argument("--sw_file", type = str, default = '', 
      help = "subword model or file")

  parser.add_argument("--output", type = str, default = "vectors.txt",
      help = "output word embedding file")
  parser.add_argument("--save_model", type = str, default = "model.pth",
      help = "model file to save")
  parser.add_argument("--load_model", type = str, default = None,
      help = "model file to load")

  parser.add_argument("--validate", action = 'store_true', default = False,
      help = "whether to validate on dev set") 
  parser.add_argument("--word_list", type = str, 
      default = '/mnt/hdd/yz568/data/word_similarity/de/2.mws353.de.txt.word',
      help = "save only words in word list")
  # params
  parser.add_argument("--size", type = int, default = 300,
      help = "word embedding dimension")
  parser.add_argument("--cbow", type = int, default = 0,
      help = "1 for cbow, 0 for skipgram")
  parser.add_argument("--window", type = int, default = 5,
      help = "context window size")
  parser.add_argument("--sample", type = float, default = 1e-5,
      help="subsample threshold")
  parser.add_argument("--negative", type = int, default = 5,
      help = "number of negative samples")
  parser.add_argument("--min_count", type = int, default = 5,
      help = "minimum frequency of a word")
  parser.add_argument("--iter", type = int, default = 5, 
      help = "number of iterations")
  parser.add_argument("--lr", type = float, default = -1.0,
      help = "initial learning rate")
  parser.add_argument("--batch_size", type = int, default = 512, 
      help = "(max) batch size")
  parser.add_argument("--thread", type = int, default = 8, 
      help = "number of thread")
  parser.add_argument("--opt", type = str, default = 'adagrad',
      help = "optimizer to use")
  parser.add_argument("--cuda", action = 'store_true', default = False, 
      help = "enable cuda")

  args = parser.parse_args()
  args.cuda = args.cuda if torch.cuda.is_available() else False
  return args
