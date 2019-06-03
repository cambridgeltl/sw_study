# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  05/15/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch.optim as optim


def get_optimizer(m):
  if m.opt == 'adagrad':
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, m.model.parameters()), lr = m.lr)
  else:
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, m.model.parameters()), lr = m.lr)
  return optimizer


def schedule_lr(m, optimizer, prog):
  lr = m.lr * (1 - prog)
  if lr < 0.0001 * m.lr:
    lr = 0.0001 * m.lr
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr
