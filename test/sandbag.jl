workspace()
using JukaiNLP
using JukaiNLP: Tokenization, Tagging
using HDF5

x = CoNLL.read("C:/Users/hshindo/Dropbox/tagging/wsj_22-24.conll")
