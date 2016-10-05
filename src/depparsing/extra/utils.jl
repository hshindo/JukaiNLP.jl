
macro template(arr...)
    feat = filter(item->item.head == :tuple, arr[1].args)
    nfeat = length(feat)
    res = :(Int[])
    for i = 1:nfeat
        val = :( tohash(len, $(i), $(feat[i].args...)); )
        push!(res.args, val)
    end
    res
end

function tohash(len::Int, arr::Int...)
    res = 1
    for i in arr
        res += i
        res += (res << 10)
        res $= (res >> 6)
    end
    res += (res << 3)
    res $= (res >> 11)
    res += (res << 15)
    abs(res) % len + 1
end
