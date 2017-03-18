
using Merlin

type MomentumSGD
    rate::Float64
    mu::Float64 # momentum coefficient
    momentum

    function MomentumSGD(rate, mu)
        new(rate, mu)
    end
end

function update!{T}(opt::MomentumSGD, value::Array{T}, grad::Array{T})
    if isdefined(opt, :momentum)
        opt.momentum = BLAS.scal(length(grad), T(opt.mu), grad, 1)
    else
        BLAS.axpy!(T(opt.mu), grad, opt.momentum)
    end
    BLAS.axpy!(-T(opt.rate), opt.momentum, value)
    fill!(grad, T(0))
end
