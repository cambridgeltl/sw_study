# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Updated 07/19/2018 
random sample sentences from wikidata
"""

#************************************************************
# Imported Libraries
#************************************************************
import sys
import random

random.seed(1234)

# one sentence per line
in_file = sys.argv[1]
# number of lines to be sampled
n = int(sys.argv[2])
with open(in_file, 'r') as fin:
  lines = fin.readlines()
  file_n = len(lines)
  if n > file_n:
    print('total lines of the input file < {}'.format(n))
  else:
    sub_lines = random.sample(lines, k = n)
    print(''.join(sub_lines).strip())
