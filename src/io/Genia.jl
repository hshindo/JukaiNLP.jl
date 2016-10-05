module Genia

using LightXML

type Token
    word
    cat
end

Token(xelem::XMLElement) = Token(content(xelem), attribute(xelem, "c"))

type Sentence
    tokens::Vector{Token}
end

Sentence(xelem::XMLElement) = Sentence(map(Token, get_elements_by_tagname(xelem, "w")))

type Article
    bibliomisc
    title::Sentence
    abst::Vector{Sentence}
end

Base.show(io::IO, a::Article) = println(io, "Article($(a.title))")

function Article(xnode::XMLElement)
    xbib = get_elements_by_tagname(xnode, "articleinfo")[1]
    xbib = get_elements_by_tagname(xbib, "bibliomisc")[1]
    xtitle = get_elements_by_tagname(xnode, "title")[1]
    xtitle = get_elements_by_tagname(xtitle, "sentence")[1]
    xabst = get_elements_by_tagname(xnode, "abstract")[1]
    xabst = get_elements_by_tagname(xabst, "sentence")
    Article(content(xbib), Sentence(xtitle), Sentence[])
end

function readxml(path)
    xdoc = parse_file(path)
    xroot = root(xdoc)
    doc = map(Article, get_elements_by_tagname(xroot, "article"))
    doc
end

end
