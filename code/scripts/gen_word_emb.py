# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated 01/18/2019
Generate word embeddings
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import sentencepiece as spm
import numpy as np

from sw.swbase_input_data import SwBaseInputData
from sms.sms_input_data import SmsInputData
from charn.charn_input_data import CharnBaseInputData

import pdb


def create_args():
  parser = argparse.ArgumentParser(description = 'Generate word embeddings from file')
  # Experiment Settings
  parser.add_argument("--compf", type = str, required = True,
      choices = ['add', 'wpadd', 'ppadd', 'mpadd', 
                 'wwadd', 'wwwpadd', 'wwppadd', 'wwmpadd',
                 'att', 'wpatt', 'ppatt', 'mpatt', 
                 'wwatt', 'wwwpatt', 'wwppatt', 'wwmpatt',
                 'mtxatt', 'wpmtxatt', 'ppmtxatt', 'mpmtxatt', 
                 'wwmtxatt', 'wwwpmtxatt', 'wwppmtxatt', 'wwmpmtxatt'],
      help = "experiment settings")
  parser.add_argument("--lang", type = str, required = True,
      help = "language")
  parser.add_argument("--emb_dim", type = int, default = 300,
      help = "embedding dimensions")
  parser.add_argument('--subword', type = str, required = True, 
      choices = ['sms', 'bpe', 'morf', 'charn'],
      help = "subword unit settings")
  # input file, one word per line
  parser.add_argument("--in_file", type = str, required = True,
      help = "input files")
  parser.add_argument("--sw_file", type = str, default = '-',
      help = "subword model or file")
  parser.add_argument("--emb_model", type = str, required = True,
      help = "subword embedding model")
  parser.add_argument("--dict_file", type = str, required = True,
      help = "model dictionary file")

  args = parser.parse_args()
  return args


def main(args):
  emb_dict, emb_model = init_model(args)
  # get the splitter for subwords 
  splitter = init_splitter(args)

  gen_emb(args, emb_dict, emb_model, splitter)


def init_model(args):
  print('Initializing current models...')
  print('Loading external data dict from: {}...'.format(args.dict_file))
  ext_dict = torch.load(args.dict_file)
  ext_dict['mo2idx'] = {(k if isinstance(k, str) else k.decode('utf-8')) : v for k, v in ext_dict['mo2idx'].items()}
  print('Loading external model from: {}...'.format(args.emb_model))
  ext_model = torch.load(args.emb_model)
  ext_model.cpu()
  ext_model.eval()
  print('Subword Size: {}'.format(len(ext_dict['mo2idx'])))
  print('Number of embeddings: {}'.format(ext_model.u_embeddings.num_embeddings - 1))
  assert(len(ext_dict['mo2idx']) == ext_model.u_embeddings.num_embeddings - 1)

  return ext_dict, ext_model


def init_splitter(args):
  if args.subword == 'morf' or args.subword == 'sms':
    splitter = read_sw_file(args.sw_file)
  elif args.subword == 'bpe':
    splitter = spm.SentencePieceProcessor()
    splitter.Load(args.sw_file)
  elif args.subword == 'charn':
    splitter = CharnBaseInputData.char_ngram_generator
  return splitter 


def read_sw_file(sw_file):
  splitter = {}
  with open(sw_file, 'r') as fin:
    for line in fin:
      linevec = line.strip().split()
      w = linevec[0]
      morphvec = linevec[1:]
      assert(w not in splitter)
      splitter[w] = morphvec
  return splitter


def gen_emb(args, emb_dict, emb_model, splitter):
  word_set = get_word_set(args)
  sw_gen_emb(args, emb_dict, emb_model, splitter, word_set)


def get_word_set(args):
  with open(args.in_file, 'r') as fin:
    word_set = fin.readlines()
  word_set = [w.strip().lower() for w in word_set]
  return word_set


def sw_gen_emb(args, emb_dict, emb_model, splitter, word_set):
  if args.subword == 'morf' or args.subword == 'sms':
    assert(len(word_set) == len(splitter.keys()))
  # morph padding
  mopad = len(emb_dict['mo2idx'])
  out_file = '{}.{}.{}.txt'.format(os.path.basename(args.in_file), args.subword, args.compf)
  ww, args.compf = SwBaseInputData.if_add_word(args.compf)

  with open(out_file, 'w') as fout:
    fout.write('{} {}\n'.format(len(word_set), args.emb_dim))
    for w in word_set:
      aff_idx = []
      morphvec = get_morphvec(args, w, splitter, ww)
      morph_idx = torch.LongTensor([emb_dict['mo2idx'][m] if m in emb_dict['mo2idx'] else mopad for m in morphvec])
      if args.subword == 'sms':
        pos_idx = SmsInputData.get_pos_idx(args.compf, morphvec)
      else:
        pos_idx = SwBaseInputData.get_pos_idx(args.compf, morphvec)
      if args.compf.startswith('pp') or args.compf.startswith('mp'):
        pos_idx = torch.LongTensor([min(pos, emb_model.u_position_embeddings.padding_idx) for pos in pos_idx])
      write_emb(args, fout, w, morph_idx, pos_idx, emb_model)


def get_morphvec(args, w, splitter, ww):
  if args.subword == 'bpe':
    morphvec = splitter.EncodeAsPieces(w.encode('utf-8'))
    morphvec = [m if isinstance(m, str) else m.decode('utf-8') for m in morphvec]
  elif args.subword == 'charn':
    morphvec = splitter(w)
  else:
    morphvec = splitter[w][:]

  morphvec = SwBaseInputData.get_morphvec(w, morphvec, args.subword, args.compf, ww)

  if args.subword == 'sms':
    morphvec = SmsInputData.update_morphvec(args.compf, morphvec)

  return morphvec


def write_emb(args, fout, w, morph_idx, aff_idx, emb_model):
  # subword embeddings
  sub_emb_u = emb_model.u_embeddings(morph_idx).unsqueeze(0)
  # pp/mp
  if args.compf.startswith('pp') or args.compf.startswith('mp'):
    aff_emb = emb_model.u_position_embeddings(aff_idx).unsqueeze(0)
    if args.compf.startswith('pp'): 
      sub_emb_u += aff_emb
    elif args.compf.startswith('mp'):
      sub_emb_u *= aff_emb

  if args.compf.endswith('att'):
    # attention
    emb_u, _ = emb_model.cal_word_embedding(sub_emb_u, torch.Tensor([1] * sub_emb_u.shape[1]), args.compf)
  else:
    # summation
    emb_u = sub_emb_u.sum(1)

  emb_u = emb_u.squeeze().data.numpy()
  emb_u_str = ' '.join([str(e) for e in emb_u])
  fout.write('{} {}\n'.format(w, emb_u_str))


if __name__ == '__main__':
  args = create_args()
  main(args)
