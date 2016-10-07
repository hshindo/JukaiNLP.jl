type POSModel
    wordfun
    charfun
    sentfun
end

function POSModel(wordfun)
    T = Float32
    x = Var()
    y = window(x, (5,), pad=(2,))
    y = Embedding(T,100,10)(y)
    y = Linear(T,50,50)(y)
    y = max(y, 2)
    charfun = Graph(y, x)

    w = Var()
    c = Var()
    y = concat(1, w, c)
    y = window(y, (150,5), pad=(0,2))
    y = Linear(T,750,300)(y)
    y = relu(y)
    y = Linear(T,300,45)(y)
    sentfun = Graph(y, (w,c))

    #=
    charfuns = [Embedding(T,100,10), Linear(T,50,50)]
    charfun = @graph begin
        x = charfuns[1](:x)
        x = window2d(x, 10,5,1,1,0,2)
        x = charfuns[2](x)
        x = max(x,2)
        x
    end
    =#
    #=
    sentfuns = [Linear(T,750,300), Linear(T,300,45)]
    sentfun = @graph begin
        x = concat(1, :wordmat, :charmat)
        x = window2d(x, 150,5,1,1,0,2)
        x = sentfuns[1](x)
        x = relu(x)
        x = sentfuns[2](x)
        x
    end
    =#
    POSModel(wordfun, charfun, sentfun)
end

function (m::POSModel)(tokens::Vector{Token})
    word_vec = map(t -> t.word, tokens)
    wordvec = reshape(word_vec, 1, length(word_vec))
    wordmat = m.wordfun(Var(wordvec))

    charvecs = map(tokens) do t
        charvec = reshape(t.chars, 1, length(t.chars))
        m.charfun(Var(charvec))
    end
    charmat = concat(2, charvecs)
    m.sentfun(wordmat, charmat)
end
