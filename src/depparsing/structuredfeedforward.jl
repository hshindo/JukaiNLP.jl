
# used in beamsearch
import Base.(<)
(<)(x::Var, y::Var) = (<)(x.data[1], y.data[1])

type StructuredFeedForward
    feedforward::FeedForward
    logits::Var
    beamid::Int
end

function initmodel!(parser::DepParser, model::Type{StructuredFeedForward})
    parser.model = StructuredFeedForward(parser.model, Var([0f0]), 1)
end

@compat function (m::StructuredFeedForward){T}(ss::AbstractVector{State{T}}, istrain=true)
    m.logits = m.feedforward(ss, istrain)
    m.beamid = 1
end

@compat function (m::StructuredFeedForward){T}(s::State{T}, act::Int)
    s.score + m.logits[[act], [m.beamid]]
end

typealias Chart Vector{Vector{State{Labeled}}}

function earlystep(golds::Chart, preds::Chart, beamsize::Int; alive=length(golds))
    zerov = Var([0f0])
    step = 0
    for i = 1:min(length(golds), alive)
        step = i
        goldscore = golds[i][1].score
        predscore = beamsize > length(preds[i]) ? zerov : preds[i][beamsize].score
        goldscore < predscore && break
    end
    step
end

function earlyupdate!(opt::SGD, golds::Vector{Chart}, preds::Vector{Chart}, alive::Vector{Int}, beamsize::Int, batchsize::Int)
    logits = Array{Var}(0)
    for (i, (gold, pred)) in enumerate(zip(golds, preds))
        k = earlystep(gold, pred, beamsize, alive=alive[i])
        k == 1 && continue
        predscores = [s.score for s in pred[k][1:min(length(pred[k]), beamsize)]]
        while length(predscores) < beamsize
            push!(predscores, Param([0f0]))
        end
        mat = concat(1, gold[k][1].score, predscores...)
        mat = reshape(mat, length(mat), 1)
        push!(logits, mat)
    end
    update!(opt, ones(Int, length(logits)), concat(2, logits))
end

import TransitionParser.beamsearch
function beamsearch{T}(parser::DepParser, ss::AbstractVector{T}, beamsize::Int, expand::Function; istrain=true)
    lessthan{T}(x::T, y::T) = x.score > y.score
    charts = Vector{Vector{T}}[]
    for (i, s) in enumerate(ss)
        push!(charts, [])
        push!(charts[i], [s])
    end

    alive = zeros(Int, length(charts))
    k = 1
    while any(v -> v == k-1, alive)
        offset = 0
        parser.model(vcat([chart[k] for chart in charts]...), istrain)
        for cid = 1:length(charts)
            cid > 1 && ( offset += length(charts[cid-1][k]) )
            chart = charts[cid]
            states = chart[k]
            if isfinal(states[1])
                push!(chart, states)
            else
                alive[cid] = k
                nexts = map(1:length(states)) do sid
                         parser.model.beamid = offset + sid
                         expand(states[sid])
                end
                nexts = vcat(nexts...)
                sort!(nexts, lt=lessthan)
                push!(chart, nexts[1:min(length(nexts), beamsize)])
            end
        end
        k += 1
    end
    charts, alive
end

function train!{T}(::Type{StructuredFeedForward}, parser::DepParser{T}, trainsents::Doc,
    testsents::Doc=Vector{Token}[]; beamsize=31, batchsize=32, iter=20,
    progbar=true, opt=SGD(0.0005, 0.9), evaliter=100, outfile="parser.dat")
    info("WILL RUN $iter ITERATIONS")

    saver = ModelSaver(outfile)
    initmodel!(parser, StructuredFeedForward)
    labelsize = targetsize(parser.model.feedforward)

    info("OPTIMIZER: ", typeof(opt))
    info("LABEL SIZE: $labelsize")
    info("BEAM SIZE: $beamsize")
    info("#TRAINSENTS: $(length(trainsents))")

    for i = 1:iter
        info("ITER $i TRAINING")
        batch = sub(trainsents, rand(1:length(trainsents), batchsize))
        loss = 0.0
        ss = map(s -> State(s, parser), batch)
        golds, _ = beamsearch(parser, ss, 1 ,expandgold)
        preds, alive = beamsearch(parser, ss, beamsize, expandpred)
        loss = earlyupdate!(opt, golds, preds, alive, beamsize, batchsize)
        res = map(chart -> chart[end][1], preds)

        info(now())
        info("LOSS: ", loss)
        evaluate(parser, res)

        if i % evaliter == 0 && !isempty(testsents)
            println()
            info("**ITER $i TESTING**")
            res = decode(StructuredFeedForward, parser, testsents,
                             beamsize=beamsize, batchsize=batchsize)
            res = convert(Array{State{T}}, res)
            uas, las = evaluate(parser, res)
            saver(parser, uas)
        end
        println()
        opt.rate *= 0.96 # step-wise decay
    end
end

function decode{T}(::Type{StructuredFeedForward}, parser::DepParser{T}, sents::Doc;
    beamsize=16, batchsize=32, progbar=true)
    res = []
    batches = 1:batchsize:length(sents)
    progbar && ( p = Progress(length(batches), 1, "", 50) )
    for i in batches
        progbar && next!(p)
        batch = sub(sents, i:min(i+batchsize-1, length(sents)))
        ss = map(s -> State(s, parser), batch)
        preds, _ = beamsearch(parser, ss, beamsize, expandpred, istrain=false)
        append!(res, map(chart -> chart[end][1], preds))
        preds = 0 # gc
    end
    res
end

