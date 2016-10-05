
# calculates unlabeled attachment score
function evaluate(parser::DepParser{Unlabeled}, ss::Vector{State{Unlabeled}})
    ignore = map(w -> parser.words[w], ["''", ",", ".", ":", "``"])
    num, den = 0, 0
    for s in ss
        pred = heads(s)
        gold = map(t -> t.head, s.tokens)
        for i in 1:length(pred)
            s.tokens[i].word in ignore && continue
            den += 1
            pred[i] == gold[i] && (num += 1)
        end
    end
    uas = num / den
    @printf "UAS: %1.4f\n" uas
    uas, 0.0
end

# calculates both unlabeled & labeled attachment score
function evaluate(parser::DepParser{Labeled}, ss::Vector{State{Labeled}})
    ignore = map(w -> parser.words[w], ["''", ",", ".", ":", "``"])
    unum, lnum, den = 0, 0, 0
    for s in ss
        predheads  = heads(s)
        predlabels = labels(s)
        goldheads  = map(t -> t.head, s.tokens)
        goldlabels = map(t -> t.label, s.tokens)
        for i in 1:length(s.tokens)
            s.tokens[i].word in ignore && continue
            den += 1
            if predheads[i] == goldheads[i]
                unum += 1
                if predlabels[i] == goldlabels[i]
                    lnum += 1
                end
            end
        end
    end
    uas = unum / den
    las = lnum / den
    @printf "UAS: %1.4f\n" uas
    @printf "LAS: %1.4f\n" las
    uas, las
end

function evaluate(parser::DepParser{Unlabeled}, ss::Vector{State{Unlabeled}}, labels::Vector{Vector{Int}})
    ignore = map(w -> parser.words[w], ["''", ",", ".", ":", "``"])
    unum, lnum, den = 0, 0, 0
    for j in 1:length(ss)
        s = ss[j]
        predheads  = heads(s)
        predlabels = labels[j]
        goldheads  = map(t -> t.head, s.tokens)
        goldlabels = map(t -> t.label, s.tokens)
        for i in 1:length(s.tokens)
            s.tokens[i].word in ignore && continue
            den += 1
            if predheads[i] == goldheads[i]
                unum += 1
                if predlabels[i] == goldlabels[i]
                    lnum += 1
                end
            end
        end
    end
    uas = unum / den
    las = lnum / den
    @printf "UAS: %1.4f\n" uas
    @printf "LAS: %1.4f\n" las
    uas, las
end
