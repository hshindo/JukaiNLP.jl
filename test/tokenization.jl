workspace()
using JukaiNLP
using JukaiNLP.Tokenization

# setup tokenizer
dirpath = Pkg.dir("JukaiNLP")
t = Tokenizer()

# training
#path = "$(dirpath)/corpus/mini-training-set.conll"
path = "$(dirpath)/.corpus/train_conll_pos_tokenized.txt"
traindata = CoNLL.read(path, 2, 5)
path = "$(dirpath)/.corpus/test_conll_pos_tokenized.txt"
testdata = CoNLL.read(path, 2, 5)
#traindata = data[1:Int(floor(length(data)*0.8))]
#testdata = data[length(traindata)+1:end]
Tokenization.train(t, 30, traindata, testdata)

#modelpath = "C:/Users/hshindo/Desktop/tokenizer.h5"
#h5save(modelpath, t.model)
#dictpath = "C:/Users/hshindo/Desktop/tokenizer.jld"
#JLD.save(dictpath, "tokenizer", t.dict)

# testing
#model = h5load(modelpath)
#dict = JLD.load(dictpath, "tokenizer")
#t = Tokenizer(dict, model)
#str = "Pierre Vinken, 61 years old, will join the board.\nI have a pen.\n"
#t(str)
