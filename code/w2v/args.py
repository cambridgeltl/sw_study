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
import argparse
import torch


def create_w2v_parser():
  # create a new parser
  parser = argparse.ArgumentParser()

  # The following arguments are mandatory
  # input text file
  parser.add_argument("--train", type = str, required = True, 
      help = "training file")

  # output vector text file
  parser.add_argument("--output", type = str, default = "vectors.txt",
      help = "output word embedding file")
  # output pytorch model file
  parser.add_argument("--save_model", type = str, default = "model.pth",
      help = "model file to save")
  # input model file
  parser.add_argument("--load_model", type = str, default = None,
      help = "model file to load")

  # embedding dim
  parser.add_argument("--size", type = int, default = 300,
      help = "word embedding dimension")
  # cbow: 1, skipgram: 0
  parser.add_argument("--cbow", type = int, default = 0,
      help = "1 for cbow, 0 for skipgram")
  # dynamic window size
  parser.add_argument("--window", type = int, default = 5,
      help = "context window size")
  # subsampling threashold
  parser.add_argument("--sample", type = float, default = 1e-5,
      help="subsample threshold")
  # negative sampling numbers
  parser.add_argument("--negative", type = int, default = 5,
      help = "number of negative samples")
  # word min count 
  parser.add_argument("--min_count", type = int, default = 5,
      help = "minimum frequency of a word")
  # training epochs
  parser.add_argument("--iter", type = int, default = 5, 
      help = "number of iterations")
  # initial learning rate
  parser.add_argument("--lr", type = float, default = 0.025,
      help = "initial learning rate")
  # gradient clipping value
  parser.add_argument("--gclip", type = float, default = 0,
      help = "gradient clipping value")
  # batch_size
  parser.add_argument("--batch_size", type = int, default = 512, 
      help = "(max) batch size")
  # optimizer
  parser.add_argument("--opt", type = str, default = 'adagrad',
      help = "optimizer to use")
  # if use cuda
  parser.add_argument("--cuda", action = 'store_true', default = False, 
      help = "enable cuda") 
  return parser


def add_mul_args(parser):
  # thread number
  parser.add_argument("--thread", type = int, default = 8, 
      help = "number of thread")
  return parser


def parse_args(parser):
  args = parser.parse_args()
  args.cuda = args.cuda if torch.cuda.is_available() else False
  return args
