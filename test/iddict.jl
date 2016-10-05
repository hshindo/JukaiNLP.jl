import JukaiNLP.IdDict

dict = IdDict()
id1 = push!(dict, "abc")
id2 = push!(dict, "def")
id3 = push!(dict, "abc")
@test id1 == 1
@test id2 == 2
@test id1 == id3
@test dict["abc"] == 1

@test length(dict) == 2

@test count(dict, id1) == 2
@test count(dict, id2) == 1

@test getkey(dict, id1) == "abc"
@test getkey(dict, id2) == "def"

@test get(dict, "abc") == 1
@test get(dict, "abd") == 0
