module OntoNotes

type Sentence
    plain::UTF8String
    tokens::Vector{UTF8String}
    tree
    leaves::Vector{UTF8String}
end

#import Base.show
#show(io::IO, s::Sentence) = show(s.plain, s.tokens, s.tree, s.leaves

"Read .onf file and convert it to Sentence."
function readonf(path)
    text = open(readall, path)
    reg = r"Plain sentence:\n-+?\n(.+?)Treebanked sentence:\n-+?\n(.+?)Tree:\n-+?\n(.+?)Leaves:\n-+?\n(.+?)\n\n"s
    map(eachmatch(reg, text)) do m
        length(m.captures) == 4 || error("invalid format")
        plain = strip(m.captures[1])
        tokens = split(strip(m.captures[2]))
        tree = format_tree(m.captures[3])
        leaves = split(m.captures[4], '\n', keep=false)
        leaves = map(strip, leaves)
        Sentence(plain, tokens, tree, leaves)
    end
end

function format_tree(s)
    s = replace(s, "\n", "")
    s = replace(s, r"\s+\(", "(")
    s = replace(s, " ", "_") # concat preterminal and terminal
    s = replace(s, ",", "comma")
    s = replace(s, ".", "period")
    s = replace(s, "-", "_")
    s
end

end
