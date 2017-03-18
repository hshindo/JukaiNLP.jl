
typealias Doc Vector{Vector{Token}}

function train!(::Type{Perceptron}, parser::DepParser, trainsents::Doc,
    testsents::Doc=Vector{Token}[]; beamsize=10, iter=20, progbar=true)
    # initialize parser.model
    initmodel!(parser, Perceptron)

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
            update!(parser.model, gold, pred)
            pred[end][1]
        end
        evaluate(parser, res)

        if !isempty(testsents)
            info("ITER $i TESTING")
            res = decode(Perceptron, parser, testsents)
            evaluate(parser, res)
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
