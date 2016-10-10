function flatten(data::Vector)
    res = Int[]
    for x in data
        append!(res, x)
    end
    res
end

function train(t::Tagger, nepochs::Int, traindata::Vector, testdata::Vector)
    train_x, train_y = encode(t, traindata)
    test_x, test_y = encode(t, testdata)

    info("# words: $(length(t.word_dict))")
    info("# chars: $(length(t.char_dict))")
    info("# tags: $(length(t.tag_dict))")

    opt = SGD(0.001)
    for epoch = 1:nepochs
        opt.rate = 0.0075 / epoch

        println("epoch: $(epoch)")
        loss = fit(train_x, train_y, t.model, crossentropy, opt)
        println("loss: $(loss)")

        test_z = Int[]
        for x in test_x
            z = t.model(x).data
            pred = argmax(z, 1)
            append!(test_z, pred)
        end
        acc = accuracy(flatten(test_y), test_z)

        println("test acc.: $(acc)")
        println("")
    end
end

function encode(t::Tagger, doc::Vector)
    data_x, data_y = Vector{Token}[], Vector{Int}[]
    for sent in doc
        push!(data_x, Token[])
        push!(data_y, Int[])
        for items in sent
            word, tag = items[1], items[2]
            word0 = replace(word, r"[0-9]", '0') |> lowercase
            #wordid = push!(t.word_dict, word0)

            wordid = get(t.word_dict, word0, 1) # experimental

            chars = Vector{Char}(word)
            charids = map(c -> push!(t.char_dict,string(c)), chars)
            tagid = push!(t.tag_dict, tag)
            token = Token(wordid, charids)
            push!(data_x[end], token)
            push!(data_y[end], tagid)
        end
    end
    data_x, data_y
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
