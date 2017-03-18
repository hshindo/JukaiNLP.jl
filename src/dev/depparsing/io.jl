
function state2array(s::State)
    st = s
    res = State[]
    while !isnull(st.prev)
        unshift!(res, st)
        st = st.prev
    end
    unshift!(res, st)
    res
end

function stack2array(s::State)
    st = s
    res = Int[]
    while !isnull(st.left)
        unshift!(res, st.top)
        st = st.left
    end
    unshift!(res, st.top)
    res
end

# prints [ a/NN b/VB ][ c/NN d/PP ]
function Base.print(io::IO, s::State)
    stack = map(stack2array(s)) do i
        i == 0 ? "ROOT/ROOT" :
        getkey(s.parser.words, s.tokens[i].word) * "/" *
        getkey(s.parser.tags, s.tokens[i].tag)
    end
    stack = join(stack, " ")
    buffer = map(s.right:length(s.tokens)) do i
        getkey(s.parser.words, s.tokens[i].word) * "/" *
        getkey(s.parser.tags, s.tokens[i].tag)
    end 
    buffer = join(buffer, " ")
    print(io, "[", stack, "][", buffer, "]")
end

function actstr(s::State{Labeled})
    act = s.prevact
    labels(id) = getkey(s.parser.labels, id)
    return acttype(act) == SHIFT ? "SHIFT" :
           acttype(act) == REDUCEL ? "REDUCE-lEFT($(labels((tolabel(act)))))" :
           acttype(act) == REDUCER ? "REDUCE-RIGHT($(labels((tolabel(act)))))" :
           throw("Invalid action: $(act).")
end

function actstr(s::State{Unlabeled})
    act = s.prevact
    return acttype(act) == SHIFT ? "SHIFT" :
           acttype(act) == REDUCEL ? "REDUCE-LEFT" :
           acttype(act) == REDUCER ? "REDUCE-RIGHT" :
           throw("Invalid action: $(act).")
end

function stacktrace{T}(io::IO, s::State{T})
    ss = state2array(s)
    for i in 2:length(ss)
        i > 1 && println(io, actstr(ss[i]))
        print(io, ss[i])
        println(io)
    end
end
stacktrace(s::State) = stacktrace(STDOUT, s)

function toconll(io::IO, s::State)
    pred = heads(s)
    for i in 1:length(s.tokens)
        t = s.tokens[i]
        items = [i, getkey(s.parser.words, t.word), "-",
                    getkey(s.parser.tags, t.tag), pred[i],
                    t.head, getkey(s.parser.labels, t.label)]
        println(io, join(items, "\t"))
    end
    println(io, "")
end
toconll(s::State) = toconll(STDOUT, s)

