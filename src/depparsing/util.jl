
# linear function with [-0.01, 0.01] uniform distribution
function myLinear(T::Type, indim::Int, outdim::Int)
    r = T(0.01)
    w = rand(-r, r, outdim, indim)
    b = fill(T(0), outdim, 1)
    Linear(zerograd!(Var(w)), (zerograd!(Var(b))))
end

function scale!(embed::Embedding)
    mean = sum(w -> sum(w.data), embed.ws)
    std = sum(w -> dot(w.data, w.data), embed.ws)
    count = length(embed.ws) * length(embed.ws[1].data)
    mean = mean / count
    std = sqrt(std / count - mean^2)
    info("Scaling word embeddings:")
    info("(mean = $mean, std = $std) -> (mean = 0.0, std = 1.0)")
    for w in embed.ws
        w.data -= mean
        w.data /= std
    end
end

function update!(opt::Union{SGD,MyAdaGrad}, gold::Vector{Int}, pred::Var)
    l = crossentropy(gold, pred) * (1 / length(gold))
    loss = sum(l.data)
    vars = gradient!(l)
    for v in vars
        if isa(v.f, Merlin.Functor)
            # l2 normalization
            BLAS.axpy!(Float32(10e-8), v.data, v.grad)
            Merlin.update!(v.f, opt)
        end
    end
    loss
end

