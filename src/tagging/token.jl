type Token
    word::Int
    chars::Vector{Int}
end

function setunkown(data::Vector{Vector{Token}}, unk::Int)
    count = mapreduce(length, +, data)
    rs = rand(count)
    i = 1
    map(data) do toks
        map(toks) do t
            word = rs[i] < 0.001 ? unk : t.word
            i += 1
            Token(word, t.chars)
        end
    end
end
