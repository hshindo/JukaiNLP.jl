export Segmenter

using Merlin

type Segmenter
    xdict::Dict{String,Int}
    ydict::Dict{String,Int}
    nn
end

function Segmenter()
    Segmenter(Dict{String,Int}(), Dict{String,Int}(), Model(4))
end

function (seg::Segmenter)(str::String)
    ids = map(collect(str)) do c
        id = seg.dict_x[string(c)]
        #id < 0 && throw("Invalid character: $(c).")
        id
    end
    x = reshape(ids, 1, length(ids))
    z = seg(Var(x))
    map(id -> getkey(seg.dict_y,id), z)
end

type Model
    g::Graph
end

function Model(outsize::Int)
    T = Float32
    x = Var([1,2])
    h = Lookup(T,300,10)(x)
    h = window(h, (70,), pads=(30,), strides=(10,))
    h = Linear(T,70,70)(h)
    h = tanh(h)
    h = Linear(T,70,outsize)(h)
    g = Graph([x], [h])
    Model(g)
end

function (m::Model)(x::Var, y=nothing)
    z = m.g(x)
    y == nothing ? argmax(z,1) : crossentropy(y,z)
end

function train(seg::Segmenter, trainpath::String, nepochs::Int)
    train_x, train_y = read(trainpath, seg.xdict, seg.ydict)
    n = floor(Int, length(train_x)*0.9)
    train_x, test_x = train_x[1:n], train_x[n+1:end]
    train_y, test_y = train_y[1:n], train_y[n+1:end]
    info("# training samples: $(length(train_x))")
    info("# test samples: $(length(test_x))")
    info("# x: $(length(seg.xdict))")
    info("# y: $(length(seg.ydict))")

    opt = SGD(0.0005)
    for epoch = 1:nepochs
        println("epoch: $epoch")
        loss = fit(train_x, train_y, seg.nn, opt)
        println("loss: $loss")

        ys = cat(1, map(y -> vec(y.data), test_y)...)
        zs = cat(1, map(x -> vec(seg.nn(x).data), test_x)...)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        println("test accuracy: $acc")
        println()
    end
end

function read(path::String, xdict, ydict)
    data_x, data_y = Var[], Var[]
    buf_x, buf_y = Int[], Int[]
    lines = open(readlines, path)
    for i = 1:length(lines)
        line = chomp(lines[i])
        if isempty(line) || i == length(lines)
            push!(data_x, Var(reshape(buf_x,1,length(buf_x))))
            push!(data_y, Var(reshape(buf_y,1,length(buf_y))))
            buf_x = Int[]
            buf_y = Int[]
        else
            items = split(line, "\t")
            id = get!(xdict, items[1], length(xdict)+1)
            push!(buf_x, id)
            id = get!(ydict, items[2], length(ydict)+1)
            push!(buf_y, id)
        end
    end
    data_x, data_y
end
