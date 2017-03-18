type Tokenizer
    dict::IdDict
    tagset::Tagset
    model
end

function Tokenizer()
    dict = IdDict(String["UNKNOWN", " ", "\n"])
    T = Float32
    x = GraphNode()
    x = window(x, (7,), pad=3)
    x = GraphNode(constant, x)
    x = Embedding(T,300,10)(x)
    x = Linear(T,70,70)(x)
    x = relu(x)
    x = Linear(T,70,3)(x)
    f = compile(x)
    Tokenizer(dict, IOE(), f)
end

function (t::Tokenizer)(chars::Vector{Char})
    unk = t.dict["UNKNOWN"]
    x = map(chars) do c
        get(t.dict, string(c), unk)
    end
    y = t.model(x).data
    tags = argmax(y,1)
    decode(t.tagset, tags)
end
(t::Tokenizer)(str::String) = t(Vector{Char}(str))
