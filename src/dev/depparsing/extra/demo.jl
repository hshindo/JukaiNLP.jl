
ccall(:jl_exit_on_sigint, Void, (Cint,), 0)

push!(LOAD_PATH, "..")
using JukaiNLP: DepParser, Perceptron, Unlabeled, Labeled, FeedForward, UnlabeledFeedForward
using JukaiNLP: readconll, train!, decode, evaluate, toconll, initmodel!, MyAdaGrad
using JukaiNLP.DepParsing: Token, readconll_withprobs
using Merlin
# using JLD
using DocOpt

doc = """shift-reduce parser

Usage:
    demo.jl train (--labeled | --unlabeled) --nn --worddict <worddict> <train_path> <model_path> [<test_path>] [--iter=<iter>] [--embedfile=<embed_file>] [--batchsize=<batch>] [--evaliter=<eval>] [--nonlinearity=<nonlinear>] [--hiddenlayers=<hidden>] [--embedlayers=<embedlayers>] [--optimizer=<opt>] [--learnrate=<rate>] [--momentum=<momentum>] [--tagdict=<tagdict>] [--train-tag-probs=<trainprobs>] [--test-tag-probs=<testprobs>] [--use-topktags=<k>]
    demo.jl train (--labeled | --unlabeled) --struct-nn <train_path> <local_model> <model_path> [<test_path>] [--iter=<iter>] [--batchsize=<batch>] [--evaliter=<eval>] [--beamsize=<beam>] [--nonlinearity=<nonlinear>] [--optimizer=<opt>] [--learnrate=<rate>] [--momentum=<momentum>] [--tagdict=<tagdict>] [--train-tag-probs=<trainprobs>] [--test-tag-probs=<testprobs>] [--use-topktags=<k>]
    demo.jl train (--labeled | --unlabeled) --perceptron --worddict <worddict> <train_path> <model_path> [<test_path>] [--iter=<iter>]

    demo.jl test <test_path> <model_path>
    demo.jl (<sent> | -) <model_path>

Options:
    <train_path>    path to CoNLL format file to use for training
    <test_path>     path to CoNLL format file to use for evaluation
    <model_path>    path to output resulting parser model

"""
    # --iter          number of training iteration
    # --embedfile     path to pretrained embedding file (used in --nn)
    # --batchsize     batch size (used in --nn)
    # --evaliter      run evaluation on test data every after this number of iteration

args = docopt(doc)

worddict = args["<worddict>"]
parsertype = args["--labeled"] ? Labeled :
             args["--unlabeled"] ? Unlabeled :
             nothing
iter = args["--iter"] != nothing ? parse(Int, args["--iter"]) :
       args["--perceptron"] ? 20 :
       args["--nn"] ? 20000 : nothing
embedfile = args["--embedfile"] != nothing ? args["--embedfile"] : ""
trainfile = args["<train_path>"]
testfile = args["<test_path>"]
modelpath = args["<model_path>"]
localmodel = args["<local_model>"]
tagfile = args["--tagdict"]
trainprob = args["--train-tag-probs"]
testprob = args["--test-tag-probs"]
topk = args["--use-topktags"] != nothing ? parse(Int, args["--use-topktags"]) : 1
batchsize = args["--batchsize"] != nothing ? parse(Int, args["--batchsize"]) : 10000
beamsize = args["--beamsize"] != nothing ? parse(Int, args["--beamsize"]) : 32
evaliter = args["--evaliter"] != nothing ? parse(Int, args["--evaliter"]) : 100
nonlinear = args["--nonlinearity"] == nothing ? relu :
            args["--nonlinearity"] == "relu" ? relu :
            args["--nonlinearity"] == "tanh" ? tanh :
            args["--nonlinearity"] == "sigmoid" ? sigmoid :
            throw("no support for $(args["--nonlinearity"])")

hiddenlayers = args["--hiddenlayers"] == nothing ? [1024] :
            map(x -> parse(Int, x), split(args["--hiddenlayers"], "x"))
embedlayers = args["--embedlayers"] == nothing ? [50,50,50] :
            map(x -> parse(Int, x), split(args["--embedlayers"], "x"))

learnrate = args["--learnrate"] != nothing ? parse(Float64, args["--learnrate"]) : 0.001
momentum = args["--momentum"] != nothing ? parse(Float64, args["--momentum"]) : 0.9

opt = args["--optimizer"] == nothing ? SGD(learnrate, momentum=momentum) :
            lowercase(args["--optimizer"]) == "sgd" ? SGD(learnrate, momentum=momentum) :
            lowercase(args["--optimizer"]) == "adagrad" ? MyAdaGrad(learnrate) :
            throw("no support for $(args["--optimizer"])")
if args["train"]

    if args["--nn"]
        parser = DepParser(parsertype, worddict)
        # trainsents = readconll_withprobs(parser, tagfile, trainfile, trainprob, topk=topk)
        # testsents = testfile == nothing ? Vector{Token}[] :
        # readconll_withprobs(parser, tagfile, testfile, testprob, topk=topk, train=false)
        trainsents = readconll(parser, trainfile)
        testsents = testfile == nothing ? Vector{Token}[] :
                        readconll(parser, testfile, train=false)
        train!(FeedForward, parser, trainsents, testsents, embed=embedfile,
                nonlinear=nonlinear, embedsizes=embedlayers, hiddensizes=hiddenlayers,
                topktags=topk!=1, opt=opt, iter=iter, batchsize=batchsize,
                evaliter=evaliter, outfile=modelpath)

    elseif args["--struct-nn"]
        parser = open(deserialize, localmodel)
        topk = typeof(parser.model.tag_f) == Linear ? -1 : 1
        trainsents = readconll_withprobs(parser, tagfile, trainfile, trainprob, topk=topk)
        testsents = testfile == nothing ? Vector{Token}[] :
        readconll_withprobs(parser, tagfile, testfile, testprob, topk=topk, train=false)
        train!(StructuredFeedForward, parser, trainsents, testsents, beamsize=beamsize,
                opt=opt, iter=iter, batchsize=batchsize, evaliter=evaliter, outfile=modelpath)

    elseif args["--perceptron"]
        parser = DepParser(parsertype, worddict)
        trainsents = readconll(parser, trainfile)
        testsents = testfile == nothing ? Vector{Token}[] :
                        readconll(parser, testfile)
        train!(Perceptron, parser, trainsents, testsents,
            iter=iter, outfile=modelpath)
    end

elseif args["test"]
    # parser = load(args["<model_path>"], "parser")
    parser = open(deserialize, modelpath)
    res = parser(testfile)
    for s in res
        toconll(s)
    end
    evaluate(parser, res)

elseif args["<sent>"] == "-"
    throw("yet to be supported")

elseif args["<sent>"] != nothing
    throw("yet to be supported")
end


