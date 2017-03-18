import .ArcStd: State, expand_gold, expand_pred, toheads

type DepParser
end

const num_feats = 1 << 26
const weights = [fill(0.0,num_feats) for i=1:3]

function tohash(data::Vector{Int})
    hash = 1
    for x in data
        hash += x
        hash += (hash << 10)
        hash $= (hash >> 6)
    end
    hash += (hash << 3)
    hash $= (hash >> 11)
    hash += (hash << 15)
    abs(hash) % num_feats + 1
end

function scorefun(s::State, acts::Vector{Int})
    feats = Int[]
    tokens = s.tokens
    s1 = tokens[s.top]
    s2 = s.left == nothing ? nulltoken : tokens[s.left.top]
    b1 = s.right <= length(tokens) ? tokens[s.right] : nulltoken
    b2 = s.right+1 <= length(tokens) ? tokens[s.right+1] : nulltoken
    xs = Any[
    Int[1, s1.formid],
    Int[2, s1.catid],
    Int[3, s1.formid, s2.catid],
    Int[4, s2.formid],
    Int[5, s2.catid],
    Int[6, s2.formid, s2.catid],
    Int[7, b1.formid],
    Int[8, b1.catid],
    Int[9, b1.formid, b1.catid],
    Int[10, s1.formid, s2.formid],
    Int[11, s1.catid, s2.catid],
    Int[12, s1.catid, b1.catid],
    Int[13, s1.formid, s1.catid, s2.catid],
    Int[14, s1.catid, s2.formid, s2.catid],
    Int[15, s1.formid, s2.formid, s2.catid],
    Int[16, s1.formid, s1.catid, s2.formid],
    Int[17, s1.formid, s1.catid, s2.formid, s2.catid],
    Int[18, s1.catid, b1.catid, b2.catid],
    Int[19, s2.catid, s1.catid, b1.catid],
    Int[20, s1.formid, b1.catid, b2.catid],
    Int[21, s2.catid, s1.formid, b1.catid],
    ]
    for x in xs
        push!(feats, tohash(x))
    end
    s.feats = feats

    scores = Float64[]
    for a in acts
        s = mapreduce(f -> weights[a][f], +, feats)
        push!(scores, s)
    end
    scores
end

function train_gold(s::State)
    feats::Vector{Int} = s.prev.feats
    for f in feats
        weights[s.prevact][f] += 1.0
    end
end

function train_pred(s::State)
    feats::Vector{Int} = s.prev.feats
    for f in feats
        weights[s.prevact][f] -= 1.0
    end
end

function decode(p::DepParser, tokens::Vector{Token})
end

function train(p::DepParser, traindata, testdata)
    for epoch = 1:20
        println("epoch: $(epoch)")
        loss = 0.0
        for i in randperm(length(traindata))
            #for x in traindata
            x = traindata[i]
            sy = State(x, scorefun)
            sz = State(x, scorefun)
            y = beamsearch(sy, 1, expand_gold)[end][1]
            z = beamsearch(sz, 1, expand_pred)[end][1]
            loss += z.score - y.score
            max_violation!(y, z, train_gold, train_pred)
        end
        println("loss: $(loss)")
        golds, preds = Int[], Int[]
        for x in testdata
            s = State(x, scorefun)
            z = beamsearch(s, 1, expand_pred)[end][1]
            append!(golds, map(t -> t.headid, x))
            append!(preds, toheads(z))
        end
        acc = accuracy(golds, preds)
        println("test acc: $(acc)")
        println("")
    end
end
