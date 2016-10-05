type Tagger
    word_dict::IdDict
    char_dict::IdDict
    tag_dict::IdDict
    model
end

function Tagger(filename::String)
    words = h5read(filename, "str")
    w = h5read(filename, "vec")
    wordfun = Embedding(w)
    word_dict = IdDict(words)
    char_dict = IdDict(String["UNKNOWN","="])
    model = POSModel(w)
    Tagger(word_dict, char_dict, IdDict(), model)
end

function (t::Tagger)(words::Vector)
    unkword = t.word_dict["UNKNOWN"]
    unkchar = t.char_dict["UNKNOWN"]
    tokens = map(words) do word
        word0 = replace(word, r"[0-9]", '0') |> lowercase
        wordid = get(t.word_dict, word0, unkword)
        chars = Vector{Char}(word)
        charids = map(c -> get(t.char_dict,string(c),unkchar), chars)
        Token(wordid, charids)
    end
    tokens = Vector{Token}(tokens)
    y = t.model(tokens).data
    tags = argmax(y, 1)
    map(x -> getkey(t.tag_dict,x), tags)
end
