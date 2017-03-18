

macro test(cond)
    teststr = string(cond)
    quote
        println("TEST: ", $teststr)
        $(esc(cond)) ? println(" -> OK") : throw("Error")
    end
end

push!(LOAD_PATH, "src")
using JukaiNLP: Perceptron, DepParser, readconll, Unlabeled, Labeled
using JukaiNLP.DepParsing: expandgold, State, isfinal, tokenat, stacktrace, tolabel, initmodel!, labelat, print, stacktrace
using TransitionParser: beamsearch

parser = DepParser(Labeled, "dict/en-word_nyt.dict")
sents = readconll(parser, "corpus/mini-training-set.conll")
initmodel!(parser, Perceptron)
s = beamsearch(State(sents[4], parser), 1, expandgold)[end][1]
stacktrace(s)

word(w) = getkey(parser.words, w)
label(l) = getkey(parser.labels, l)

@test isfinal(s)
@test tokenat(s, s.top).word == 2
# @show word(tokenat(s, s.rchild).word)
@test word(tokenat(s, s.rchild).word) == "says"
@test word(tokenat(s, s.rchild.lchild).word) == "raziq"
@test word(tokenat(s, s.rchild.lchild.lchild).word) == "border"
@test word(tokenat(s, s.rchild.lchild.lsibl.lchild).word) == "police"
@test word(tokenat(s, s.rchild.lchild.lsibl.lsibl.lchild).word) == "commander"
@test word(tokenat(s, s.rchild.lchild.lsibl.lsibl.lsibl.lchild).word) == "abdul"
@test word(tokenat(s, s.rchild.rchild).word) == "."
@test word(tokenat(s, s.rchild.rsibl.rchild).word) == "found"
@test word(tokenat(s, s.rchild.rsibl.rchild.rchild).word) == "in"


@test label(labelat(s, s.rchild)) == "ROOT"
@test label(labelat(s, s.rchild.lsibl)) == "nsubj"
@test label(labelat(s, s.rchild.lsibl.left.lsibl)) == "nn" # border
@test label(labelat(s, s.rchild.lsibl.left.lsibl.lsibl)) == "nn" # police
@test label(labelat(s, s.rchild.lsibl.left.lsibl.lsibl.lsibl)) == "nn" # commander
@test label(labelat(s, s.rchild.lsibl.left.lsibl.lsibl.lsibl.lsibl)) == "nn" # abdul
@test word(tokenat(s, s.rchild.lsibl.left.lsibl.lsibl.lsibl.lsibl.left).word) == "abdul" # abdul
@test label(labelat(s, s.rchild.rchild)) == "punct"
@test label(labelat(s, s.rchild.rsibl.rchild)) == "ccomp" #found
@test label(labelat(s, s.rchild.rsibl.rchild.rchild)) == "prep" #in

