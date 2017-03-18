module TokenizationSentence

importall ..JukaiNLP
using Merlin

export train, Tokenizer

include("tokenizer.jl")
include("train.jl")

end
