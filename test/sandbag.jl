workspace()

path = "C:/Users/hshindo/Dropbox/Shared/ConceptDep/deim2016/n-sada/web1gram"
lines = open(readlines, path)
webdict = Dict()
map(lines) do line
    line = chomp(line)
    items = split(line, '\t')
    webdict[items[1]] = parse(Int, items[2])
end

path = "D:/wiktionary_dict_clean_unique.tsv"
lines = open(readlines, path)
out = []
for line in lines
    line = chomp(line)
    items = split(line, '\t')
    w = items[1]
    haskey(webdict, w) || continue
    push!(out, line)
end
open("out.txt", "w") do fp
    for o in out
        println(fp, o)
    end
end
