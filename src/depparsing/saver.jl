
type ModelSaver
    file
    bestscore
end

ModelSaver(file) = ModelSaver(file, 0.0)

function (s::ModelSaver)(model, score)
    if s.bestscore < score
        println("\nSAVER: BEST SCORE $score")
        print("SAVER: SAVING MODEL TO $(s.file)... ")
        open(io -> serialize(io, model), s.file, "w")
        s.bestscore = score
        println("DONE")
    end
end
