
type LabelerFF
    word_f
    tag_f
    label_f
    nonlinear
    W
end

const none = 46

function setheads!{T}(s::State{T})
    @assert isfinal(s)
    s.head = nullstate(T)
    stack = [s]
    while !isempty(stack)
        s = pop!(stack)
        if !isnull(s.lchild)
            s.lchild.head = s
            push!(stack, s.lchild)
        end
        st = s
        while !isnull(st.lsibl.lchild)
            st.lsibl.lchild.head = s
            push!(stack, st.lsibl.lchild)
            st = st.lsibl
        end
        if !isnull(s.rchild)
            s.rchild.head = s
            push!(stack, s.rchild)
        end
        st = s
        while !isnull(st.rsibl.rchild)
            st.rsibl.rchild.head = s
            push!(stack, st.rsibl.rchild)
            st = st.rsibl
        end
    end
end

type LExample
    w::Vector{Int}
    t::Union{Matrix{Float32},Vector{Int}}
    l::Vector{Int}
    target::Int
end

function gen_trainexample(s::State)
    sent = s.tokens
    labelsize = targetsize(s.parser.labeler)
    w, t = genfeatures(s)
    l = Int[] # [label(s0 -> s0head), label(s0head -> s0head's head)]
    push!(l, sent[s.top].label)
    push!(l, tokenat(s, s.head).label)
    push!(l, tokenat(s, s.head.rchild).label)
    push!(l, tokenat(s, s.head.lchild).label)
    push!(l, tokenat(s, s.head.head.rchild).label)
    push!(l, tokenat(s, s.head.head.lchild).label)
    target = sent[s.top].label
    LExample(w, t, l, target)
end

targetsize(m::LabelerFF) = size(m.W[end].w.data)[1]

function initmodel!(parser::DepParser, labeler::Type{LabelerFF}, embed,
    topktags, nonlinear, sparsesizes, embedsizes, hiddensizes; inherit=false)
    T = Float32
    if inherit
        word_f = Embedding(copy(parser.model.word_f.ws), IntSet())
        tag_f  = Embedding(copy(parser.model.tag_f.ws), IntSet())
        label_f = Embedding(copy(parser.model.label_f.ws), IntSet())
    else
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
        label_f = Embedding(T, length(parser.labels), embedsizes[2])
    end
    indim = sum(sparsesizes .* embedsizes)
    outdim = length(parser.labels) - 1 # exclude NONE
    W = [myLinear(T, indim, hiddensizes[1]),
         myLinear(T, hiddensizes[1], outdim)]
    parser.labeler = LabelerFF(word_f, tag_f, label_f, nonlinear, W)
    info("INPUT: [S^word,S^tag,S^label] = ", sparsesizes)
    info("EMBED DIMS: [word,tag,label] = ", embedsizes)
    info("HIDDEN LAYER: ", hiddensizes)
    info("OUTPUT DIM: ", outdim)
    info("NONLINEAR: ", nonlinear)
end

# TODO: make State have id field
# to tell where the State is in a batch
# called from expand(::State ::Int)
@compat function (m::LabelerFF){T}(s::State{T}, act::Int)
    Var([0f0])
end

@compat function (m::LabelerFF)(batch::AbstractVector, istrain=true)
    wordvec, tagvec, labelvec = [], [], []
    for s in batch
        push!(wordvec, s[1])
        push!(tagvec, s[2])
        push!(labelvec, s[3])
    end
    wordmat = m.word_f(Var(hcat(wordvec...)))
    tagmat = m.tag_f(Var(hcat(tagvec...)))
    labelmat = m.label_f(Var(hcat(labelvec...)))
    x = concat(1, wordmat, tagmat, labelmat)
    x = m.nonlinear(m.W[1](x))
    x = dropout(x, 0.5, istrain)
    x = m.W[end](x)
    x
end

function genfeatures(s::State)
    # word, tag
    s0     = tokenat(s, s.top)
    s0l    = tokenat(s, s.lchild)
    s0ll1  = tokenat(s, s.lchild.lchild)
    s0ll2  = tokenat(s, s.lchild.lsibl.lchild)
    s0lr1  = tokenat(s, s.lchild.rchild)
    s0lr2  = tokenat(s, s.lchild.rsibl.rchild)
    s0l2   = tokenat(s, s.lsibl.lchild)
    s0l2l1 = tokenat(s, s.lsibl.lchild.lchild)
    s0l2l2 = tokenat(s, s.lsibl.lchild.lsibl.lchild)
    s0l2r1 = tokenat(s, s.lsibl.lchild.rchild)
    s0l2r2 = tokenat(s, s.lsibl.lchild.rsibl.rchild)
    s0l3   = tokenat(s, s.lsibl.lsibl.lchild)
    s0r    = tokenat(s, s.rchild)
    s0rr1  = tokenat(s, s.rchild.rchild)
    s0rr2  = tokenat(s, s.rchild.rsibl.rchild)
    s0rl1  = tokenat(s, s.rchild.lchild)
    s0rl2  = tokenat(s, s.rchild.lsibl.lchild)
    s0r2   = tokenat(s, s.rsibl.rchild)
    s0r2r1 = tokenat(s, s.rsibl.rchild.rchild)
    s0r2r2 = tokenat(s, s.rsibl.rchild.rsibl.rchild)
    s0r2l1 = tokenat(s, s.rsibl.rchild.lchild)
    s0r2l2 = tokenat(s, s.rsibl.rchild.lsibl.lchild)
    s0r3   = tokenat(s, s.rsibl.rsibl.rchild)

    s0h   = tokenat(s, s.head)
    s0h2  = tokenat(s, s.head.head)
    s0hl  = tokenat(s, s.head.lchild)
    s0hr  = tokenat(s, s.head.rchild)
    s0h2l  = tokenat(s, s.head.head.lchild)
    s0h2r  = tokenat(s, s.head.head.rchild)

    words = [s0.word, s0l.word, s0ll1.word, s0ll2.word, s0lr1.word, s0lr2.word,
             s0l2.word, s0l2l1.word, s0l2l2.word, s0l2r1.word, s0l2r2.word, s0l3.word,
             s0.word, s0r.word, s0rr1.word, s0rr2.word, s0rl1.word, s0rl2.word,
             s0r2.word, s0r2r1.word, s0r2r2.word, s0r2l1.word, s0r2l2.word, s0r3.word,
             s0h.word, s0h2.word, s0hl.word, s0hr.word, s0h2l.word, s0h2r.word]

    tags = [s0.tag, s0l.tag, s0ll1.tag, s0ll2.tag, s0lr1.tag, s0lr2.tag,
             s0l2.tag, s0l2l1.tag, s0l2l2.tag, s0l2r1.tag, s0l2r2.tag, s0l3.tag,
             s0.tag, s0r.tag, s0rr1.tag, s0rr2.tag, s0rl1.tag, s0rl2.tag,
             s0r2.tag, s0r2r1.tag, s0r2r2.tag, s0r2l1.tag, s0r2l2.tag, s0r3.tag,
             s0h.tag, s0h2.tag, s0hl.tag, s0hr.tag, s0h2l.tag, s0h2r.tag]


    # words = [s0.word, s0l.word, s0l2.word, s0l3.word, s02l.word, s02l2.word, s0r.word,
    #  s0r2.word, s0r3.word, s02r.word, s02r2.word, s0h.word, s0h2.word]
    # tags = [s0.tag, s0l.tag, s0l2.tag, s0l3.tag, s02l.tag, s02l2.tag, s0r.tag,
    #         s0r2.tag, s0r3.tag, s02r.tag, s02r2.tag, s0h.tag, s0h2.tag,]
    return words, tags
end

# function getchildren(heads::Array{Int})
#     res = map(_ -> [], heads)
#     for (c, h) in enumerate(heads)
#         push!(res[h], c)
#     end
#     res
# end

# function head2state{T}(s::State{T})
#     @assert isfinal(s) && s.top == 0
#     res = Array(State{T}, length(s.tokens))
#     while !isnull(s.prev)
#         s = s.prev
#         if s.top != 0 && !isdefined(res, s.top)
#             res[s.top] = s
#         end
#     end
#     res
# end

function assign!{T}(labeler::LabelerFF, ss::Vector{State{T}})
    # predicted labels
    blabels = map(s ->fill(none, length(s.tokens)), ss)
    bheads  = map(s -> heads(s), ss)
    stack = map(i -> (ss[i].prev ,i), 1:length(ss)) # (batches i'th tree, batchid)
    while !isempty(stack)
        bsize = min(length(stack), 10000)
        batch, stack = stack[1:bsize], stack[bsize+1:end]
        bfeat = map(batch) do tup
            s, bid = tup
            w, t = genfeatures(s)
    push!(l, tokenat(s, s.head.rchild).label)
    push!(l, tokenat(s, s.head.lchild).label)
    push!(l, tokenat(s, s.head.head.rchild).label)
    push!(l, tokenat(s, s.head.head.lchild).label)
            l = [blabels[bid][s.top],
                get(blabels[bid], bheads[bid][s.top], none),
                s.head.rchild.top != 0 ? blabels[bid][bheads[bid][s.head.rchild.top]] : none,
                s.head.lchild.top != 0 ? blabels[bid][bheads[bid][s.head.lchild.top]] : none,
                s.head.head.rchild.top != 0 ? blabels[bid][bheads[bid][s.head.head.rchild.top]] : none,
                s.head.head.lchild.top != 0 ? blabels[bid][bheads[bid][s.head.head.lchild.top]] : none]
            (w, t, l)
        end
        preds = argmax(labeler(bfeat, false).data, 1)
        for (i, (s, bid)) in enumerate(batch)
            blabels[bid][s.top] = preds[i]
            !isnull(s.lchild) && push!(stack, (s.lchild, bid))
            st = s
            while !isnull(st.lsibl.lchild)
                push!(stack, (st.lsibl.lchild, bid))
                st = st.lsibl
            end
            !isnull(s.rchild) && push!(stack, (s.rchild, bid))
            st = s
            while !isnull(st.rsibl.rchild)
                push!(stack, (st.rsibl.rchild, bid))
                st = st.rsibl
            end
        end
    end
    @assert all(la -> all(v -> v != none, la), blabels)
    blabels
end
# if  && blabels[bid][s.(arc).top] == none

# function assign!{T}(labeler::LabelerFF, ss::Vector{State{T}})
#     @assert all(isfinal, ss)
#     # predicted labels
#     blabels   = map(s -> fill(none, length(s.tokens)), ss)
#     bstates   = map(s -> head2state(s), ss) # no State with s.top == 0
#     bheads    = map(s -> heads(s), ss)
#     bchildren = map(heads -> getchildren(heads), bheads)
#     stack = map(i -> (1,i), 1:length(ss)) # (headid, batchid)
#     while !isempty(stack)
#         bsize = min(length(stack), length(ss))
#         batch, stack = stack[1:bsize], stack[bsize:end]
#         bfeat = map(batch) do tup
#             hid, bid = tup
#             s = bstates[bid][hid]
#             w, t = genfeatures(s)
#             l = [labels[bid][s.top],
#                  heads[bid][s.top] != 0 ? labels[bid][heads[bid][s.top]] : 1]
#             (w, t, l)
#         end
#         preds = argmax(parser.labeler(bfeat, false).data, 1)
#         for (i, (hid, bid)) in enumarate(batch)
#             blabels[bid][hid] = preds[i]
#             for childid in bchildren
#                 push!(stack, (childid, bid))
#             end
#         end
#     end
#     blabels
# end
#
typealias Doc Vector{Vector{Token}}
function train!{T}(::Type{LabelerFF}, parser::DepParser{T}, trainsents::Doc,
    testsents::Doc=Vector{Token}[]; embed="", batchsize=32, iter=20, progbar=true,
    nonlinear=tanh, sparsesizes=[30,30,6] ,embedsizes=[50,50,50], hiddensizes=[200],
    topktags=false, opt=MyAdaGrad(0.01), evaliter=100, outfile="parser.dat")
    info("WILL RUN $iter ITERATIONS")

    if !isdefined(parser, :model)
        initmodel!(parser, FeedForward, embed, topktags,
               nonlinear, sparsesizes, embedsizes, hiddensizes)
    end
    saver = ModelSaver(outfile)
    initmodel!(parser, LabelerFF, embed, topktags,
               nonlinear, sparsesizes, embedsizes, hiddensizes)

    projectivize!(trainsents, false)

    trainsamples = LExample[]
    oracletrees = State{T}[]
    for sent in trainsents
        s = State(sent, parser)
        exs = Array(State{T}, length(sent))
        # exs[i] should be the final State object
        # at which i'th token appeared at s0.
        while !isfinal(s)
            s = expandgold(s)[1]
            if s.top != 0
                exs[s.top] = s
            end
        end
        setheads!(s)
        push!(oracletrees, s)
        @assert all(i -> isdefined(exs, i), 1:length(exs))
        append!(trainsamples, map(gen_trainexample, exs))
    end
    @assert all(isfinal, oracletrees)
    samplesize = length(trainsamples)
    labelsize = targetsize(parser.labeler)

    i = 1
    testtrees = deepcopy(testsents)
    projectivize!(testtrees)
    testtrees = map(testtrees) do sent
        print("\r", i)
        i += 1
        s = State(sent, parser)
        while !isfinal(s)
            s = expandgold(s)[1]
        end
        s
    end

    info("OPTIMIZER: ", typeof(opt))
    info("BATCH SIZE: $batchsize LABEL SIZE: $labelsize")
    info("#BATCHES: $(div(samplesize, batchsize))")
    info("#SAMPLES: $(samplesize)")

    for i = 1:iter
        batch = sub(trainsamples, rand(1:samplesize, batchsize))
        preds = parser.labeler(map(b -> (b.w, b.t, b.l), batch))
        golds = map(s -> s.target, batch)
        correct = reduce(0, zip(argmax(preds.data, 1), golds)) do v, tup
            v + Int(tup[1] == tup[2])
        end
        accuracy = correct / batchsize
        loss = update!(opt, golds, preds)
        info("Epoch: $i\tLoss: $loss\tAcc: $accuracy")

        if i % evaliter == 0 && !isempty(testsents)
            info("Epoch: $i\tEvaluating labeler")
            @time res = assign!(parser.labeler, testtrees)
            res = convert(Vector{Vector{Int}}, res)
            uas, las = evaluate(parser, testtrees, res)

            info("Epoch: $i\tEvaluating pipeline")
            @time tree, labels = decode(LabelerFF, parser, testsents)
            uas, las = evaluate(parser, tree, labels)
            saver(parser, las) # decides whether to save based on LAS
        end
    end
end

# do pipeline job: unlabeled parser -> labeler
function decode{T}(::Type{LabelerFF}, parser::DepParser{T}, sents::Doc;
    batchsize=32, progbar=true)
    batches = 1:batchsize:length(sents)
    blabels = []
    res = []
    for k in batches
        batch = k:min(k+batchsize-1, endof(sents))
        ss = map(s -> State(s, parser), sub(sents, batch))
        map(setheads!, ss)
        parsegreedy!(parser, ss)
        labels = assign!(parser.labeler, ss)
        append!(res, ss)
        append!(blabels, labels)
    end
    res, blabels
end

