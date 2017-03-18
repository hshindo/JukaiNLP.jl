
import Base.(^)
function (^){N <: Number}(x::Var, n::N)
    y = map(v -> v^n, x.data)
    function df{T}(gy::UniArray{T})
        if hasgrad(x)
            x.data += n * map(v -> v^(n-1), x.data) * gy
        end
    end
    Var(y, [x], ^, df)
end
