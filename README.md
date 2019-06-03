# Subword-informed word representations training framework
We have provided a general framework for training subword-informed word representations.

## Segmentation methods
[ChipMunk](http://cistern.cis.lmu.de/chipmunk)

[Morfessor](https://morfessor.readthedocs.io/en/latest/index.html)

[BPE](https://github.com/bheinzerling/bpemb)

## Calculate new word embeddings from subword embeddings
Call `gen_word_emb.py` to generate embeddings of new words for a specific composition function or use `batch_gen_word_emb.sh` to generate for all composition functions.

Your input, i.e. `--in_file` in input arg, needs to be a list of word, where each line only consists of a single word. 

