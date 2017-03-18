function encode(t::Tokenizer, doc::Vector)
    unk, space, lf = t.dict["UNKNOWN"], t.dict[" "], t.dict["\n"]
    chars = Int[]
    ranges = UnitRange{Int}[]
    pos = 1
    for sent in doc
        for (word,tag) in sent
            for c in tag
                c == '_' && continue
                if c == 'S' # space
                    push!(chars, space)
                elseif c == 'N' # newline
                    push!(chars, lf)
                end
                pos += 1
            end
            for c in word
                push!(chars, push!(t.dict,string(c)))
            end
            push!(ranges, pos:pos+length(word)-1)
            pos += length(word)
        end
    end
    chars, ranges
end

function flatten(data::Vector)
    res = Int[]
    for x in data
        append!(res, x)
    end
    res
end

function train(t::Tokenizer, nepochs::Int, traindata::Vector, testdata::Vector)
    function conv(data)
        chars, ranges = encode(t, data)
        tags = encode(t.tagset, ranges)
        # mini-batch
        chars_batch = [chars[i:min(i+1000-1,length(chars))] for i=1:1000:length(chars)]
        tags_batch = [tags[i:min(i+1000-1,length(tags))] for i=1:1000:length(tags)]
        chars_batch, tags_batch
    end
    train_x, train_y = conv(traindata)
    test_x, test_y = conv(testdata)

    opt = SGD(0.001)
    for epoch = 1:nepochs
        println("epoch: $(epoch)")
        loss = fit(train_x, train_y, t.model, crossentropy, opt)
        println("loss: $(loss)")

        test_z = map(test_x) do x
            argmax(t.model(x).data, 1)
        end
        y, z = flatten(test_y), flatten(test_z)
        ranges_y, ranges_z = decode(t.tagset, y), decode(t.tagset, z)
        fscore(ranges_y, ranges_z)

        acc = accuracy(flatten(test_y), flatten(test_z))

        println("test acc.: $(acc)")
        println("")
    end
end

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
