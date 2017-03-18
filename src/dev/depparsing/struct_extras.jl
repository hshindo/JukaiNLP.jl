
function embed2linear(embed::Embedding)
    w = hcat(map(v -> v.data, embed.ws)...)
    outdim = size(w, 1)
    T = typeof(w[1])
    b = fill(T(0), outdim, 1)
    Linear(Merlin.Param(w), Merlin.Param(b))
end

function indmaxk{T}(vec::Array{T}, k::Int)
    arr = copy(vec)
    res = Int[]
    while k > 0
        idx = indmax(arr)
        push!(res, idx)
        arr[idx] = -10000000f0 #typemin(T)
        k -= 1
    end
    res
end

function readprobs(parser::DepParser, tagpath::AbstractString, conllpath::AbstractString, topk::Int)
    tags = map(chomp, open(readlines, tagpath))
    tag2id = Dict([tag => id for (id, tag) in enumerate(tags)])
    doc = Vector[]
    push!(doc, [])
    for line in open(readlines, conllpath)
        line = chomp(line)
        if isempty(line)
            push!(doc, [])
            continue
        end
        items = split(line, "\t")
        word = items[1]
        probs = map(v -> parse(Float32, v), items[2:end])
        # prob(NONE) == 0.0
        probs = [probs[tag2id[getkey(parser.tags, i)]] for i = 2:length(tags)+1]
        probs = convert(Vector{Float32}, probs)
        unshift!(probs, 0f0)
        if topk == 1
            probs = indmax(probs)
        elseif topk > 1
            topidx = indmaxk(probs, topk)
            probs = [i in topidx ? probs[i] : 0f0 for i = 1:length(probs)]
            probs = reshape(probs, length(probs), 1)
        else
            probs = reshape(probs, length(probs), 1)
        end
        push!(doc[end], (lowercase(word), probs))
    end
    if topk != 1
        vsize = size(doc[1][1][2])
        global roottoken = Token(2, zeros(Float32, vsize), 0, 1)
    end
    doc
end

function readconll_withprobs(parser, tagpath, conllpath, probpath; topk=1, train=true)
    conlldoc = readconll(parser, conllpath, train=train)
    conllprobs = readprobs(parser, tagpath, probpath, topk)
    @assert length(conlldoc) == length(conllprobs)
    for (sent, probs) in zip(conlldoc, conllprobs)
        @assert length(sent) == length(probs)
        for (token, prob) in zip(sent, probs)
            form = getkey(parser.words, token.word)
            token.tag = prob[2]
        end
    end
    conlldoc
end
