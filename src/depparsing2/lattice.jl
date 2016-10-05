immutable Arc
    next::Int
end

type WFSM{T}
    states::Vector{T}
    arcs::Vector{Arc}
end
