
abstract Model

type Perceptron  <: Model
    weights::Matrix{Float64}
end

function initmodel!(parser::DepParser, model::Type{Perceptron})
    if parser.parsertype == Unlabeled
        parser.model = Perceptron(zeros(1 << 23, 3))
    elseif parser.parsertype == Labeled
        labelsize = length(parser.labels)
        labelsize == 0 && warn("labelsize 0. should readconll on train data first")
        dim = 1 + 2 * labelsize
        parser.model = Perceptron(zeros(1 << 23, dim))
    else
        throw("parsertype not supported: $(parser.parsertype)")
    end
end

@compat function (p::Perceptron)(s::State, act::Int)
    if !isdefined(s, :feat)
        s.feat = featuregen(s)
    end
    res = s.score
    for f in s.feat
        res += p.weights[f,act]
    end
    res
end

function traingold!(p::Perceptron, s::State)
    act = s.prevact
    feat = s.prev.feat
    for f in feat
        p.weights[f,act] += 1.0
    end
end

function trainpred!(p::Perceptron, s::State)
    act = s.prevact
    feat = s.prev.feat
    for f in feat
        p.weights[f,act] -= 1.0
    end
end

function featuregen(s::State)
    n0 = tokenat(s, s.right)
    n1 = tokenat(s, s.right+1)
    s0 = tokenat(s, s)
    s1 = tokenat(s, s.left)
    s2 = tokenat(s, s.left.left)
    s0l = tokenat(s, s.lchild)
    s0r = tokenat(s, s.rchild)
    s1l = tokenat(s, s.left.lchild)
    s1r = tokenat(s, s.left.rchild)

    len = size(s.parser.model.weights, 1) # used in @template macro
    @template begin
        # template (1)
        (s0.word,)
        (s0.tag,)
        (s0.word, s0.tag)
        (s1.word,)
        (s1.tag,)
        (s1.word, s1.tag)
        (n0.word,)
        (n0.tag,)
        (n0.word, n0.tag)

        # additional for (1)
        (n1.word,)
        (n1.tag,)
        (n1.word, n1.tag)

        # template (2)
        (s0.word, s1.word)
        (s0.tag, s1.tag)
        (s0.tag, n0.tag)
        (s0.word, s0.tag, s1.tag)
        (s0.tag, s1.word, s1.tag)
        (s0.word, s1.word, s1.tag)
        (s0.word, s0.tag, s1.tag)
        (s0.word, s0.tag, s1.word, s1.tag)

        # additional for (2)
        (s0.tag, s1.word)
        (s0.word, s1.tag)
        (s0.word, n0.word)
        (s0.word, n0.tag)
        (s0.tag, n0.word)
        (s1.word, n0.word)
        (s1.tag, n0.word)
        (s1.word, n0.tag)
        (s1.tag, n0.tag)

        # template (3)
        (s0.tag, n0.tag, n1.tag)
        (s1.tag, s0.tag, n0.tag)
        (s0.word, n0.tag, n1.tag)
        (s1.tag, s0.word, n0.tag)

        # template (4)
        (s1.tag, s1l.tag, s0.tag)
        (s1.tag, s1r.tag, s0.tag)
        (s1.tag, s0.tag, s0r.tag)
        (s1.tag, s1l.tag, s0.tag)
        (s1.tag, s1r.tag, s0.word)
        (s1.tag, s0.word, s0l.tag)

        # template (5)
        (s2.tag, s1.tag, s0.tag)
    end
end

typealias Doc Vector{Vector{Token}}

function train!{T}(::Type{Perceptron}, parser::DepParser{T}, trainsents::Doc,
    testsents::Doc=Vector{Token}[]; beamsize=10, iter=20, progbar=true, outfile="")

    initmodel!(parser, Perceptron)
    projectivize!(trainsents)

    outfile == "" || ( saver = ModelSaver(outfile) )
    info("LOADING SENTENCES")
    info("WILL RUN $iter ITERATIONS")
    for i = 1:iter
        info("ITER $i TRAINING")
        progbar && ( p = Progress(length(trainsents), 1, "", 50) )
        res = map(trainsents) do s
            progbar && next!(p)
            s = State(s, parser)
            gold = beamsearch(s, 1, expandgold)
            pred = beamsearch(s, beamsize, expandpred)
            max_violation!(gold[end][1], pred[end][1],
                s -> traingold!(parser.model, s),
                s -> trainpred!(parser.model, s)
            )
            pred[end][1]
        end
        evaluate(parser, res)

        if !isempty(testsents)
            info("ITER $i TESTING")
            res = decode(Perceptron, parser, testsents)
            uas, _ = evaluate(parser, res)
            outfile == "" || saver(parser, uas)
        end
    end
end

function decode(::Type{Perceptron}, parser::DepParser, sents::Doc;
    beamsize=10, progbar=true)
    progbar && ( p = Progress(length(sents), 1, "", 50) )
    map(sents) do s
        progbar && next!(p)
        pred = State(s, parser)
        beamsearch(pred, beamsize, expandpred)[end][1]
    end
end
