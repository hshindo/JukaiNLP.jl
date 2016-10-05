
type Token
    word::Int
    tag # ::Int
    head::Int
    label::Int
end

roottoken = Token(2, 1, 0, 1) # use PADDING

function normalize(word)
    w = lowercase(word)
    if w == "-lrb-"
        return "("
    elseif w == "-rrb-"
        return ")"
    else
        return w
    end
end

function readconll(parser::DepParser, path::AbstractString; train=true, cutoff=1)
    doc = Vector{Token}[]
    push!(doc, Token[])
    for line in open(readlines, path)
        line = chomp(line)
        if isempty(line)
            push!(doc, Token[])
        else
            items = split(line)
            word, tag, head, label = items[2], items[5], items[7], items[8]
            # word = replace(lowercase(word), r"\d", "0")
            word = normalize(word)
            wordid = get(parser.words, word, parser.words["UNKNOWN"])
            # cutoff
            # if !train && count(parser.words, wordid) <= cutoff
            #     wordid = parser.words["UNKNOWN"]
            # end
            tagid = train ? push!(parser.tags, tag) : parser.tags[tag]
            headid = parse(Int, head)
            labelid = train ? push!(parser.labels, label) : parser.labels[label]
            t = Token(wordid, tagid, headid, labelid)
            push!(doc[end], t)
        end
    end
    filter!(s -> length(s) > 0, doc)
    train && push!(parser.labels, "NONE")
    doc
end

function toconll(io::IO, parser::DepParser, sent::Vector{Token})
    for (id, t) in enumerate(sent)
        line = [id,
                getkey(parser.words, t.word), 
                getkey(parser.tags, t.tag),
                getkey(parser.tags, t.tag),
                "-", "-",
                t.head,
                getkey(parser.labels, t.label),
                "-", "-"]
        println(io, join(line, "\t"))
    end
    println(io)
end
toconll(parser::DepParser, sent::Vector{Token}) = toconll(STDOUT, parser, sent)

function isprojective(sent::Vector{Token})
    counter = -1
    function visittree(w::Int)
        for i = 1:w-1
            if sent[i].head == w && !visittree(i)
                return false
            end
        end
        counter += 1
        w == counter || return false
        for i = w+1:length(sent)
            if sent[i].head == w && !visittree(i)
                return false
            end
        end
        return true
    end
    visittree(0)
end

function projectivize!(doc::Vector{Vector{Token}}, discard_nonprojective=false)
    discard = Int[]
    for i = 1:length(doc)
        if !projectivize!(doc[i], discard_nonprojective)
            push!(discard, i)
        end
    end
    for i in reverse(discard)
        splice!(doc, i)
    end
end

function projectivize!(sent::Vector{Token}, discard_nonprojective=false)
    ntokens = length(sent)
    left = Vector{Int}(ntokens)
    right = Vector{Int}(ntokens)

    while true
        fill!(left, 0)
        fill!(right, ntokens+1)

        for i = 1:ntokens
            headid = sent[i].head
            l = min(i, headid)
            r = max(i, headid)

            for j = l+1:r-1
                left[j] < l && ( left[j] = l )
                right[j] > r && ( right[j] = r )
            end
        end

        deepest_arc = 0
        max_depth = 0

        for i = 1:ntokens
            headid = sent[i].head
            headid == 0 && continue
            l = min(i, headid)
            r = max(i, headid)
            lbound = max(left[l], left[r])
            rbound = min(right[l], right[r])

            if l < lbound || r > rbound
                discard_nonprojective && return false
                depth = 0
                j = i
                while j != 0
                    depth += 1
                    j = sent[j].head
                end
                if depth > max_depth
                    deepest_arc = i
                    max_depth = depth
                end
            end
        end

        deepest_arc == 0 && return true
        lifted_head = sent[sent[deepest_arc].head].head
        sent[deepest_arc].head = lifted_head
    end
end
