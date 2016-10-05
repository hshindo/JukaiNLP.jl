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
        res_x, res_y = Vector{Int}[], Vector{Int}[]
        for x in data
            chars, ranges = encode(t, x)
            tags = encode(t.tagset, ranges)
            push!(res_x, chars)
            push!(res_y, tags)
        end
        res_x, res_y
    end
    train_x, train_y = conv(traindata)
    test_x, test_y = conv(testdata)

    #opt = AdaGrad(0.01)
    opt = SGD(0.0001, momentum=0.9)
    for epoch = 1:nepochs
        println("epoch: $(epoch)")
        loss = fit(t.model, crossentropy, opt, train_x, train_y)
        println("loss: $(loss)")

        test_z = map(test_x) do x
            argmax(t.model(x).data, 1)
        end
        acc = accuracy(flatten(test_y), flatten(test_z))

        println("test acc.: $(acc)")
        println("")
    end
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
