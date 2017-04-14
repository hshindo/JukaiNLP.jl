function fscore(golds::Vector{UnitRange{Int}}, preds::Vector{UnitRange{Int}})
    g = Set(golds)
    p = Set(preds)
    inter = intersect(g, p)
    correct = length(inter)

    prec = correct / length(p)
    recall = correct / length(g)
    fvalue = 2*recall*prec/(recall+prec)
    println("precision: $(prec)")
    println("recall: $(recall)")
    println("f-value: $(fvalue)")
end

function accuracy(golds::Vector{Int}, preds::Vector{Int})
    @assert length(golds) == length(preds)
    correct = 0
    total = 0
    for i = 1:length(golds)
        golds[i] == preds[i] && (correct += 1)
        total += 1
    end
    correct / total
end
