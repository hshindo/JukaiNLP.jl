function readconf(path)
    entities = []
    lines = open(readlines, path)
    for line in lines
        line = chomp(line)
        (length(line) == 0 || line[1] =='#') && continue
        items = split(line, '\t')
        t = items[1]
        items = split(items[2], ':')
        att, val = items[1], items[2]
        e = Dict("type" => t, "labels" => [t], att => val)
        push!(entities, e)
    end
    entities
end
