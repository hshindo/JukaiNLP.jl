export DepParser

abstract ParserType
type Unlabeled <: ParserType end
type Labeled <: ParserType end

type DepParser{T <: ParserType}
    words::IdDict{String}
    tags::IdDict{String}
    labels::IdDict{String}
    parsertype::Type{T}
    model
    labeler

    function DepParser(words, tags, labels, parsertype)
        new(words, tags, labels, parsertype)
    end
end

function DepParser{T <: ParserType}(parsertype::Type{T}, path::String)
    words = load(IdDict, path)
    tags = IdDict{String}()
    labels = IdDict{String}()
    push!(tags, "NONE")
    # push!(labels, "NONE")
    DepParser{T}(words, tags, labels, parsertype)
end

function (parser::DepParser){T}(sents::Vector{Vector{T}})
    model_t = typeof(parser.model)
    decode(model_t, parser, sents)
end

function (parser::DepParser)(filepath::String)
    sents = readconll(parser, filepath, train=false)
    model_t = typeof(parser.model)
    decode(model_t, parser, sents)
end
