type Tokenizer
    dict::IdDict
    model
end

function Tokenizer()
    dict = IdDict(String["UNKNOWN", " ", "LF"])
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
    Tokenizer(dict, g)
end

function (t::Tokenizer)(chars::Vector{Char})
    unk = t.dict["UNKNOWN"]
    x = map(chars) do c
        str = c == '\n' ? "LF" : string(c)
        get(t.dict, str, unk)
    end
    y = t.model(x).data
    tags = argmax(y,1)
    decode(chars, tags)
end
(t::Tokenizer)(str::String) = t(Vector{Char}(str))

function decode(chars::Vector{Char}, tags::Vector{Int})
    length(chars) == length(tags) || throw("Length unmatch")
    doc = Vector{String}[]
    tokens = String[]
    buffer = Char[]
    pos = 1
    function push_buffer!(i::Int)
        push!(buffer, chars[i])
    end
    function push_token!()
        length(buffer) == 0 && return
        form = String(buffer)
        #t = Token(pos=pos, form=form)
        push!(tokens, form)
        pos += length(buffer) + 1
        buffer = Char[]
    end
    function push_tokens!()
        length(tokens) == 0 && return
        push!(doc, tokens)
        tokens = String[]
    end

    for i = 1:length(chars)
        t = tags[i]
        if t == 1
            push_buffer!(i)
        elseif t == 2
            continue
        elseif t == 3
            push_buffer!(i)
            push_token!()
        elseif t == 4
            push_buffer!(i)
            push_token!()
            push_tokens!()
        else
            error("Invalid tag: $(t)")
        end
    end
    push_token!()
    push_tokens!()
    doc
end
