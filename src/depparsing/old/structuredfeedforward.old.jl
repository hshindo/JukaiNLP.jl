
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

@compat function (m::StructuredFeedForward){T}(ss::AbstractVector{State{T}})
    m.logits = m.feedforward(ss)
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

function earlyupdate!(opt::SGD, golds::Chart, preds::Chart, beamsize::Int)
    k = earlystep(golds, preds, beamsize)
    k == 1 && return 0.0
    predscores = [s.score for s in preds[k][1:min(length(preds[k]), beamsize)]]
    mat = concat(1, golds[k][1].score, predscores...)
    mat = reshape(mat, length(mat), 1)
    update!(opt, [1], mat)
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
function beamsearch{T}(parser::DepParser, s::T, beamsize::Int, expand::Function)
    chart = Vector{T}[]
    push!(chart, [s])
    lessthan{T}(x::T, y::T) = x.score > y.score

    k = 1
    while k <= length(chart)
        states = chart[k]
        length(states) > beamsize && sort!(states, lt=lessthan)
        beam = 1:min(beamsize, length(states))
        parser.model(states[beam])
        for i in beam
            parser.model.beamid = i
            for s in expand(states[i])
                while s.step > length(chart)
                    push!(chart, T[])
                end
                push!(chart[s.step], s)
            end
        end
        k += 1
    end
    sort!(chart[end], lt=lessthan)
    chart
end

function beamsearch{T}(parser::DepParser, ss::AbstractVector{T}, beamsize::Int, expand::Function)
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
        parser.model(vcat([chart[k] for chart in charts]...))
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
    testsents::Doc=Vector{Token}[]; embed="", beamsize=31, batchsize=32, iter=20,
    progbar=true, opt=SGD(0.0001, 0.9), evaliter=100, outfile="parser.dat")
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
        # progbar && ( p = Progress(batchsize, 1, "", 50) )
        # res = map(batch) do s
        #     progbar && next!(p)
        #     s = State(s, parser)
        #     golds = beamsearch(parser, s, 1, expandgold)
        #     preds = beamsearch(parser, s, beamsize, expandpred)
        #     loss += earlyupdate!(opt, golds, preds, beamsize)
        #     preds[end][1]
        # end
        ss = map(s -> State(s, parser), batch)
        golds, _ = beamsearch(parser, ss, 1 ,expandgold)
        preds, alive = beamsearch(parser, ss, beamsize, expandpred)
        loss = earlyupdate!(opt, golds, preds, alive, beamsize, batchsize)
        res = map(chart -> chart[end][1], preds)

        info(now())
        info("LOSS: ", loss)
        evaluate(parser, res)

        if i % evaliter == 0
            info("SAVING PARSER TO $outfile")
            open(io -> serialize(io, parser), outfile * "." * string(i), "w")
            # println()
            # info("**ITER $i TESTING**")
            # res = decode(StructuredFeedForward, parser, testsents,
            #                  beamsize=beamsize, batchsize=batchsize)
            # uas, las = evaluate(parser, res)
            # saver(parser, uas)
        end
        println()
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
        preds, _ = beamsearch(parser, ss, beamsize, expandpred)
        append!(res, map(chart -> chart[end][1], preds))
        preds = 0 # gc
    end
    res
    # map(sents) do s
    #     progbar && next!(p)
    #     s = State(s, parser)
    #     beamsearch(parser, s, beamsize, expandpred)[end][1]
    # end
end

