using HttpServer
using WebSockets
using Merlin
using JukaiNLP
using JukaiNLP.TokenizationSentence
using JukaiNLP.Tagging
using JLD
using JSON
using PyCall

unshift!(PyVector(pyimport("sys")["path"]), "mitie-v0.2-python-2.7")
unshift!(PyVector(pyimport("sys")["path"]), "entity-disambi")
@pyimport NER
is_linux() && (@pyimport mymodule)

type Token
    form::String
    cat::String
    ne::Vector{Any}
    dep::Int
end

Token(form::String) = Token(form, "", [], 0)

function todict(t::Token)
    Dict("form"=>t.form, "cat"=>t.cat, "ne"=>t.ne)
end

const samples = [
    Token("Pierre", "NNP", [], 2),
    Token("Vinken", "NNP", [], 8),
    Token(",", ",", [], 2),
    Token("61", "CD", [], 5),
    Token("years", "NNS", [], 6),
    Token("old", "JJ", [], 2),
    Token(",", ",", [], 2),
    Token("will", "MD", [], 0),
    Token("join", "VB", [], 8),
    Token("the", "DT", [], 11),
    Token("board", "NN", [], 9),
    Token("Nov.", "NNP", [], 9),
    Token("29", "CD", [], 12),
    Token(".", ".", [], 8)
]
const samples2 = [ "Diamond", "Shamrock", "Offshore", "Partners","said","it",
    "had","discovered","gas","offshore","Louisiana", "."]

function load_tokenizer()
    println("loading tokenizer...")
    path = joinpath(Pkg.dir("JukaiNLP"), "web/models")
    model = h5load("$(path)/tokenizer.h5")
    dict = JLD.load("$(path)/tokenizer.jld", "tokenizer")
    Tokenizer(dict, model)
end
const tokenizer = load_tokenizer()

function load_postagger()
    println("loading postagger...")
    path = joinpath(Pkg.dir("JukaiNLP"), "web/models")
    m = h5load("$(path)/postagger.h5")
    model = Tagging.POSModel(m["wordfun"],m["charfun"],m["sentfun"])
    t = JLD.load("$(path)/postagger.jld", "postagger")
    t.model = model
    t
end
const postagger = load_postagger()

function setpostag!(tokens::Vector{Token}, forms::Vector{String})
    tags = postagger(forms)
    for i = 1:length(tokens)
        tokens[i].cat = tags[i]
    end
end

function setner!(tokens::Vector{Token}, forms::Vector{String})
    ners = NER.predict(forms)
    for (range,tag) in ners
        tokens[start(range)+1].ne = Any[tag,length(range)]
        if is_linux()
            #target = forms[start(range)+1:last(range)+1]
            #mymodule.predict(target, forms)
        end
    end
end

include("conf.jl")

const filepath = dirname(@__FILE__)
const clients = Dict()
const conf = begin
    e = readconf(joinpath(filepath,"conf/visual.conf"))
    d = Dict("entity_types" => e)
    #=
    "relation_types" => [
        Dict(
            "type" => "Anaphora",
            "labels" => ["Anaphora", "Ana"],
            "dashArray" => "3,3",
            "color" => "purple",
            "args" => [
                Dict("role" => "Anaphor", "targets" => ["Person"]),
                Dict("role" => "Entity", "targets" => ["Person"])
            ]
        )
    ]
    =#
    JSON.json(d)
end

wsh = WebSocketHandler() do req, client
    println("Client: $(client.id) is connected.")
    write(client, conf)
    while true
        #println("Request from $(client.id) recieved.")
        msg = bytestring(read(client))
        length(msg) > 1000 && continue
        chars = Vector{Char}(msg)

        doc = tokenizer(chars)
        doc = map(doc) do sent
            tokens = map(w -> Token(w), sent)
            setpostag!(tokens, sent)
            setner!(tokens, sent)
            map(todict, tokens)
        end
        res = JSON.json(doc)
        write(client, res)
    end
end

onepage = readstring(joinpath(dirname(@__FILE__), "index.html"))
httph = HttpHandler() do req::Request, res::Response
    Response(onepage)
end
server = Server(httph, wsh)
run(server, 3000)
