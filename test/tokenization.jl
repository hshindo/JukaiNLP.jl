workspace()
using Merlin
using JLD
using JukaiNLP
using JukaiNLP.TokenizationSentence

# setup tokenizer
dirpath = Pkg.dir("JukaiNLP")
t = Tokenizer()

# training
path = "$(dirpath)/.corpus/webtreebank.conll"
data = readconll(path, [2,3])
traindata = data[1:Int(floor(length(data)*0.8))]
testdata = data[length(traindata)+1:end]
TokenizationSentence.train(t, 50, traindata, testdata)
modelpath = "C:/Users/hshindo/Desktop/tokenizer.h5"
h5save(modelpath, t.model)
dictpath = "C:/Users/hshindo/Desktop/tokenizer.jld"
JLD.save(dictpath, "tokenizer", t.dict)
throw("finish")

# testing
model = h5load(modelpath)
dict = JLD.load(dictpath, "tokenizer")
t = Tokenizer(dict, model)
str = "Pierre Vinken, 61 years old, will join the board.\nI have a pen.\n"
t(str)

# setup tokenizer
dirpath = Pkg.dir("JukaiNLP")
t = Tokenizer()

# training
trainpath = "$(dirpath)/corpus/mini-training-set.conll"
traindata = readconll(trainpath, [2,11])
Tokenization.train(t, 100, [traindata], [traindata])
filename = "C:/Users/hshindo/Desktop/tokenizer.h5"
h5save(filename, t.model)

# testing
model = h5load(filename)
str = "Pierre Vinken, 61 years old, will join the board.\nI have a pen.\n"
result = t(str)
join(map(r -> str[r], result), " ")
