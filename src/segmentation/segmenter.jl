type Segmenter
    nn::Graph
end

function (s::Segmenter)(data)

end

function train(s::Segmenter, data)

end

using MLDatasets

function setup_data()
    indict = Dict{String,Int}()
    outdict = Dict{String,Int}()

    doc = CoNLL.read(".data/webtreebank.conll")
    #traindoc = doc[1:]
    #testdoc = CoNLL.read(".data/wsj_22-24.conll")
    #info("# sentences of train doc: $(length(traindoc))")
    #info("# sentences of test doc: $(length(testdoc))")

    traindata = setup_data(traindoc, worddict, chardict, tagdict)
    testdata = setup_data(testdoc, worddict, chardict, tagdict)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    info("# tags: $(length(tagdict))")
    traindata, testdata, worddict, chardict, tagdict
end

function setup_data(doc::Vector, surface_dict, label_dict)
    data = []
    for tokens in doc
        for items in tokens
            surface = items[1]
            boundary = parse(Int, items[2])
            boundary == 1 || boundary == 2 || throw("Invalid boundary: $boundary.")
            label = items[3]
            surfaceid = get!(surface_dict, surface, length(surface_dict)+1)
            labelid = get!(label_dict, label, length(label_dict)+1)
            push!(data, ())
        end

        w = Int[]
        cs = Var[]
        t = Int[]
        for items in sent
            word, tag = items[2], items[5]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)
            chars = Vector{Char}(word0)
            charids = map(c -> get!(chardict,c,length(chardict)+1), chars)
            tagid = get!(tagdict, tag, length(tagdict)+1)
            push!(w, wordid)
            push!(cs, Var(charids))
            push!(t, tagid)
        end
        x = (Var(w), cs)
        push!(data, (x,Var(t)))
    end
    data
end
