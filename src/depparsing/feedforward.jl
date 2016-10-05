
"""
A Fast and Accurate Dependency Parser using Neural Networks, Chen and Manning, EMNLP 2014
"""

type FeedForward
    word_f
    tag_f
    label_f
    nonlinear
    W
end

type Example
    wordids::Vector{Int}
    tagids::Union{Matrix{Float32},Vector{Int}}
    labelids::Vector{Int}
    target::Int
    valid::Vector{Float32}
end

function Example(s::State)
    labelsize = targetsize(s.parser.model)
    w, t, l = sparsefeatures(s)
    valid = [Float32(isvalid(s, acttype(act))) for act = 1:labelsize]
    Example(w, t, l, -1, valid)
end

targetsize(m::FeedForward) = size(m.W[end].w.data)[1]

function initmodel!(parser::DepParser, model::Type{FeedForward}, embed,
    topktags, nonlinear, sparsesizes, embedsizes, hiddensizes)
    T = Float32
    if embed == ""
        info("USING EMBEDDINGS WITH UNIFORM DISTRIBUTION [-0.01, 0.01]")
        word_f = Embedding(T, length(parser.words), embedsizes[1])
    else
        info("USING EMBEDDINGS LOADED FROM $(embed)")
        word_f = Embedding(embed, Float32)
        scale!(word_f)
    end
    if topktags
        tag_f = Linear(T, length(parser.tags), embedsizes[2])
    else
        tag_f = Embedding(T, length(parser.tags), embedsizes[2])
    end
    label_f = Embedding(T, length(parser.labels), embedsizes[3])
    indim = sum(sparsesizes .* embedsizes)
    outdim = 1 + 2 * (length(parser.labels) - 1) # exclude NONE
    W = [myLinear(T, indim, hiddensizes[1]),
         myLinear(T, hiddensizes[1], outdim)]
    parser.model = FeedForward(word_f, tag_f, label_f, nonlinear, W)
    info("INPUT: [S^word,S^tag,S^label] = ", sparsesizes)
    info("EMBED DIMS: [word,tag,label] = ", embedsizes)
    info("HIDDEN LAYER: ", hiddensizes)
    info("OUTPUT DIM: ", outdim)
    info("NONLINEAR: ", nonlinear)
end

# TODO: make State have id field
# to tell where the State is in a batch
# called from expand(::State ::Int)
function (m::FeedForward){T}(s::State{T}, act::Int)
    Var([0f0])
end

"""
Improved Transition-Based Parsing and Tagging with Neural Networks, EMNLP, 2015
k-best POS tags
"""
function tagvectors(f::Linear, tagvec)
    batchsize = length(tagvec)
    embedsize, inputsize = size(f.w)
    h = f(Var(hcat(map(v -> reshape(v, inputsize, 20), tagvec)...)))
    reshape(h, embedsize*20, batchsize)
end

function tagvectors(f::Embedding, tagvec)
    f(Var(hcat(tagvec...)))
end

function (m::FeedForward)(batch::AbstractVector{Example}, istrain=true)
    wordvec, tagvec, labelvec = [], [], []
    for s in batch
        push!(wordvec, s.wordids)
        push!(tagvec, s.tagids)
        push!(labelvec, s.labelids)
    end
    wordmat = m.word_f(Var(hcat(wordvec...)))
    tagmat = tagvectors(m.tag_f, tagvec)
    labelmat = m.label_f(Var(hcat(labelvec...)))
    x = concat(1, wordmat, tagmat, labelmat)
    x = m.nonlinear(m.W[1](x))
    x = dropout(x, 0.5, istrain)
    x = m.W[end](x)
    valids = hcat(map(x -> x.valid, batch)...)
    x.data .*= valids
    x
end

function (m::FeedForward){T}(batch::AbstractVector{State{T}}, istrain=true)
    m(map(Example, batch), istrain)
end

function sparsefeatures(s::State)
    # word, tag
    b0 = tokenat(s, s.right)
    b1 = tokenat(s, s.right + 1)
    b2 = tokenat(s, s.right + 2)
    b3 = tokenat(s, s.right + 3)
    s0 = tokenat(s, s.top)
    s0l = tokenat(s, s.lchild)
    s0l2 = tokenat(s, s.lsibl.lchild)
    s0r = tokenat(s, s.rchild)
    s0r2 = tokenat(s, s.rsibl.rchild)
    s02l = tokenat(s, s.lchild.lchild)
    s12r = tokenat(s, s.rchild.rchild)
    s1 = tokenat(s, s.left)
    s1l = tokenat(s, s.left.lchild)
    s1l2 = tokenat(s, s.left.lsibl.lchild)
    s1r = tokenat(s, s.left.rchild)
    s1r2 = tokenat(s, s.left.rsibl.rchild)
    s12l = tokenat(s, s.left.lchild.lchild)
    s12r = tokenat(s, s.left.rchild.rchild)
    s2 = tokenat(s, s.left.left)
    s3 = tokenat(s, s.left.left.left)

    # labels
    s0rc_label = labelat(s, s.rchild)
    s0rc2_label = labelat(s, s.rsibl.rchild)
    s0lc_label = labelat(s, s.lsibl)
    s0lc2_label = labelat(s, s.lsibl.lsibl)
    s02l_label = labelat(s, s.lsibl.left.lsibl)
    s02r_label = labelat(s, s.rchild.rchild)
    s1rc_label = labelat(s, s.left.rchild)
    s1rc2_label = labelat(s, s.left.rsibl.rchild)
    s1lc_label = labelat(s, s.left.lsibl)
    s1lc2_label = labelat(s, s.left.lsibl.lsibl)
    s12l_label = labelat(s, s.left.lsibl.left.lsibl)
    s12r_label = labelat(s, s.left.rchild.rchild)

    words = [b0.word, b1.word, b2.word, b3.word, s0.word, s0l.word, s0l2.word,
    s0r.word, s0r2.word, s02l.word, s12r.word, s1.word, s1l.word, s1l2.word,
    s1r.word, s1r2.word, s12l.word, s12r.word, s2.word, s3.word]

    tags = [b0.tag, b1.tag, b2.tag, b3.tag, s0.tag, s0l.tag, s0l2.tag,
    s0r.tag, s0r2.tag, s02l.tag, s12r.tag, s1.tag, s1l.tag, s1l2.tag,
    s1r.tag, s1r2.tag, s12l.tag, s12r.tag, s2.tag, s3.tag]

    labels = [s0rc_label, s0rc2_label, s0lc_label, s0lc2_label, s02l_label,
    s02r_label, s1rc_label, s1rc2_label, s1lc_label, s1lc2_label,
    s12l_label, s12r_label]

    return words, tags, labels
end

function parsegreedy!{T}(parser::DepParser{T}, ss::Vector{State{T}})
    while !all(isfinal, ss)
        preds = parser.model(ss, false)
        bestacts = argmax(preds.data, 1)
        for i = 1:length(ss)
            isfinal(ss[i]) && continue
            ss[i] = expand(ss[i], bestacts[i])
        end
    end
    ss
end

typealias Doc Vector{Vector{Token}}

function train2!{T}(::Type{FeedForward}, parser::DepParser{T}, trainsents::Doc,
    testsents::Doc=Vector{Token}[]; embed="", batchsize=10000, iter=20000,
    nonlinear=relu, sparsesizes=[20,20,12] ,embedsizes=[50,50,50], hiddensizes=[1024],
    topktags=false, opt=MyAdaGrad(0.01), evaliter=200, outfile="parser.dat")
    info("WILL RUN $iter ITERATIONS")

    saver = ModelSaver(outfile)
    initmodel!(parser, FeedForward, embed, topktags,
               nonlinear, sparsesizes, embedsizes, hiddensizes)
end

function train!{T}(::Type{FeedForward}, parser::DepParser{T}, trainsents::Doc,
    testsents::Doc=Vector{Token}[]; embed="", batchsize=10000, iter=20000,
    nonlinear=relu, sparsesizes=[20,20,12] ,embedsizes=[50,50,50], hiddensizes=[1024],
    topktags=false, opt=MyAdaGrad(0.01), evaliter=200, outfile="parser.dat")
    info("WILL RUN $iter ITERATIONS")

    saver = ModelSaver(outfile)
    initmodel!(parser, FeedForward, embed, topktags,
               nonlinear, sparsesizes, embedsizes, hiddensizes)

    projectivize!(trainsents, false)

    trainsamples = Example[]
    for s in trainsents
        s = State(s, parser)
        while !isfinal(s)
            ex = Example(s)
            s = expandgold(s)[1]
            ex.target = s.prevact
            push!(trainsamples, ex)
        end
    end
    samplesize = length(trainsamples)
    labelsize = targetsize(parser.model)

    info("OPTIMIZER: ", typeof(opt))
    info("BATCH SIZE: $batchsize LABEL SIZE: $labelsize")
    info("#BATCHES: $(div(samplesize, batchsize))")
    info("#SAMPLES: $(samplesize)")

    for i = 1:iter
        batch = view(trainsamples, rand(1:samplesize, batchsize))
        preds = parser.model(batch)
        golds = map(s -> s.target, batch)
        correct = reduce(0, zip(argmax(preds.data, 1), golds)) do v, tup
            v + Int(tup[1] == tup[2])
        end
        accuracy = correct / batchsize
        loss = update!(opt, golds, preds)
        info("Epoch: $i\tLoss: $loss\tAcc: $accuracy")

        if i % evaliter == 0 && !isempty(testsents)
            println()
            info("**ITER $i TESTING**")
            res = decode(FeedForward, parser, testsents)
            uas, las = evaluate(parser, res)
            saver(parser, uas)
            println()
        end
    end
end

function decode{T}(::Type{FeedForward}, parser::DepParser{T}, sents::Doc;
    batchsize=32, progbar=true)
    batches = 1:batchsize:length(sents)
    res = State{T}[]
    for k in batches
        batch = k:min(k+batchsize-1, endof(sents))
        ss = map(s -> State(s, parser), view(sents, batch))
        parsegreedy!(parser, ss)
        append!(res, ss)
    end
    res
end
