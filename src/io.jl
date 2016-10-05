export readconll, rawtext

function readconll(path, columns=Int[])
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

function rawtext(filename)
    strs = String[]
    lines = open(readlines, filename)
    for line in lines
        line = chomp(line)
        length(line) == 0 && continue
        items = split(line, '\t')
        tag = items[3]
        if tag != "_"
            for c in tag
                if c == 'S'
                    push!(strs, " ")
                elseif c == 'N'
                    push!(strs, "\n")
                else
                    throw("Invalid char: $(c)")
                end
            end

        end
        push!(strs, items[1])
    end
    join(strs, "")
end
