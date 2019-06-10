# Subword-informed word representation training framework
We provide a general framework for training subword-informed word representations by varying the following components: 
- [subword segmentation methods](#subword-segmentation-methods);
- [subword embeddings and position embeddings](#subword-embeddings-and-position-embeddings);
- composition functions;

For the whole framework architecture and more details, please refer to the [reference](#references).

There are 4 segmentation methods, 3 possible ways of embedding subwords, 3 ways of enhancing with position embeddings, and 3 different composition functions.

Here is a full table of different options and their labels:

| Component | Option| Label |
|---|---|---|
| Segmentation methods 	| CHIPMUNK <br> Morfessor <br> BPE <br> Character n-gram| *sms* <br> *morf* <br> *bpe* <br> *charn*| 
| Subword embeddings 	| w/o word token <br> w/ word token <br> w/ morphotactic tag (only for *sms*)| - <br> *ww* <br> *wp*| 
| Position embeddings 	| w/o position embedding <br> addition <br> elementwise multiplication| - <br> *pp* (not applicable to *wp*) <br> *mp* (not applicable to *wp*)| 
| Composition functions | addition <br> single self-attention <br> multi-head self-attention| *add* <br> *att* <br> *mtxatt*| 

For example, *sms.wwppmtxatt* means we use CHIPMUNK as segmentation, insert word token into the subword sequence, enhance with additive position embedding, and use multi-head self-attention as composition function.


## Subword segmentation methods
Taking the word *dishonestly* as an example, with different segmentation methods, the word will be segmented into the following subword sequence:
- [ChipMunk](http://cistern.cis.lmu.de/chipmunk):  (*<dis*,  *honest*, *ly>*) + (*PREFIX*,  *ROOT*, *SUFFIX*)
- [Morfessor](https://morfessor.readthedocs.io/en/latest/index.html): (*<dishonest*, *ly>*)
- [BPE](https://github.com/bheinzerling/bpemb) (10k merge ops): (*<dish*, *on*, *est*, *ly>*)
- [Character n-gram](https://aclweb.org/anthology/Q17-1010) (from 3 to 6): (*<di*, *dis*, ... , *ly>*, *<dis*, ... ,*tly>*, *<dish*, ... , *stly>*, *<disho*, ... , *estly>*)

where *<* and *>* are word start and end markers.

After the segmentation, we will obtain a subword sequence S for each segmentation method, and another morphortactic tag sequence T for *sms*.

## Subword embeddings and position embeddings
We can embed the subword sequence S directly into subword embedding sequence by looking up in the subword embedding matrix, or insert a word token (*ww*) into S before embedding, i.e. for *sms* it will be (*<dis*, *honest*, *ly>*,  *\<dishonestly\>*).

Then we can enhance the subword embeddings with additive (*pp*) or elementwise (*mp*) multiplication.

For *sms*, we can also embed the concatenation of the subword and its morphortactic tags (*wp*): (*<dis:PREFIX*, *honest:ROOT*, *ly>*:*SUFFIX*). And *\<dishonest\>:WORD* will be inserted if we choose *ww*. Note that position embeddings are not applicable to *wp* as a kind of *morphological position* information has already been provided.

## Prerequisites


## Calculate new word embeddings from subword embeddings
Call `gen_word_emb.py` to generate embeddings of new words for a specific composition function or use `batch_gen_word_emb.sh` to generate for all composition functions.

Your input, i.e. `--in_file` in input arg, needs to be a list of word, where each line only consists of a single word. 

## References
- [A Systematic Study of Leveraging Subword Information for Learning Word Representations.](https://arxiv.org/abs/1904.07994) Yi Zhu, Ivan Vulić, and Anna Korhonen. In Proc. of NAACL 2019.

