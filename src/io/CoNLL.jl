module CoNLL

function read(path::String, columns::Int...)
    doc = []
    sent = []
    lines = open(readlines, path)
    for line in lines
        line = chomp(line)
        if length(line) == 0
            length(sent) > 0 && push!(doc, sent)
            sent = []
        else
            items = split(line, '\t')
            length(items) > 0 && (items = items[columns])
            push!(sent, items)
        end
    end
    length(sent) > 0 && push!(doc, sent)
    doc
end

end
