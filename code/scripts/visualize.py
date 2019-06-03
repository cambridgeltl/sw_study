# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated 01/14/2019
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import gensim.models
import pdb

infile_x = sys.argv[1]
infile_y = sys.argv[2]
gmodel_x = gensim.models.KeyedVectors.load_word2vec_format(infile_x)
gmodel_y = gensim.models.KeyedVectors.load_word2vec_format(infile_y)
en_list = ['january', 'man', 'microsoft', 'great', 'said', 'was', 'percent', 'market', 'year']
de_list = ['januar', 'mann', 'microsoft', 'große', 'prozent', 'reuters', 'sagte', 'händler', 'habe']
for en_word in en_list:
  print(en_word)
  en_results = gmodel_x.most_similar(en_word, topn=3)
  en_rstr ='{} ({:.2f})'.format(en_results[0][0], en_results[0][1])
  for i in range(1, len(en_results)):
    en_rstr += ', {}'.format(en_results[i][0])
  print(en_rstr)
    
  de_results = gmodel_y.most_similar([gmodel_x[en_word]], topn=3)
  de_rstr ='{} ({:.2f})'.format(de_results[0][0], de_results[0][1])
  for i in range(1, len(de_results)):
    de_rstr += ', {}'.format(de_results[i][0])
  print(de_rstr)
for de_word in de_list:
  print(de_word)

  en_results = gmodel_x.most_similar([gmodel_y[de_word]], topn=3)
  en_rstr ='{} ({:.2f})'.format(en_results[0][0], en_results[0][1])
  for i in range(1, len(en_results)):
    en_rstr += ', {}'.format(en_results[i][0])
  print(en_rstr)

  de_results = gmodel_y.most_similar(de_word, topn=3)
  de_rstr ='{} ({:.2f})'.format(de_results[0][0], de_results[0][1])
  for i in range(1, len(de_results)):
    de_rstr += ', {}'.format(de_results[i][0])
  print(de_rstr)
