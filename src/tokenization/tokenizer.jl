type Tokenizer
    dict::IdDict
    tagset::Tagset
    model
end

function Tokenizer()
    dict = IdDict(String["UNKNOWN", " ", "\n"])
    T = Float32
    embed = Embedding(T, 100, 10)
    ls = [Linear(T,70,70), Linear(T,70,4)]
    g = @graph begin
        chars = identity(:chars)
        x = Var(reshape(chars,1,length(chars)))
        x = embed(x)
        x = window2d(x,10,7,1,1,0,3)
        x = ls[1](x)
        x = relu(x)
        x = ls[2](x)
        x
    end
    Tokenizer(dict, IOE(), g)
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
