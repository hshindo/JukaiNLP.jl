function flatten(data::Vector)
    res = Int[]
    for x in data
        append!(res, x)
    end
    res
end

function encode(dict::IdDict, data::Vector)
    chars = Int[]
    tags = Int[]
    for sent in data
        for items in sent
            char, tag = items[1], items[2]
            charid = push!(dict, string(char))
            push!(chars, charid)
            push!(tags, parse(Int,tag))
        end
    end
    x = Vector{Int}[]
    y = Vector{Int}[]
    for i = 1:1000:length(chars)-1000
        push!(x, chars[i:i+1000])
        push!(y, tags[i:i+1000])
    end
    x, y
end

function train(t::Tokenizer, nepochs::Int, traindata::Vector, testdata::Vector)
    train_x, train_y = encode(t.dict, traindata)
    test_x, test_y = encode(t.dict, testdata)
    info("#dict: $(length(t.dict))")

    #opt = AdaGrad(0.01)
    opt = SGD(0.0001, momentum=0.9)
    for epoch = 1:nepochs
        println("epoch: $(epoch)")
        loss = fit(t.model, crossentropy, opt, train_x, train_y)
        println("loss: $(loss)")

        test_z = map(test_x) do x
            argmax(t.model(x).data, 1)
        end
        #acc = accuracy(flatten(test_y), flatten(test_z))
        accuracy_sent(flatten(test_y), flatten(test_z))

        #println("test acc.: $(acc)")
        println("")
    end
end

function accuracy_sent(golds::Vector{Int}, preds::Vector{Int})
    @assert length(golds) == length(preds)
    correct_g, correct_p = 0, 0
    total_g, total_p = 0, 0
    for i = 1:length(golds)
        if golds[i] == 4
            golds[i] == preds[i] && (correct_g += 1)
            total_g += 1
        end
        if preds[i] == 4
            golds[i] == preds[i] && (correct_p += 1)
            total_p += 1
        end
    end
    prec = correct_g / total_g
    recall = correct_p / total_p
    println("precision: $(prec)")
    println("recall: $(recall)")
    println("f-value: $(2*recall*prec/(recall+prec))")
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
