type NN
    g::Graph
end

function NN()
    T = Float32
    x = Var()
    h = Lookup(T,300,10)(x)
    h = window(h, (70,), pads=(30,), strides=(10,))
    h = Linear(T,70,70)(h)
    h = relu(h)
    h = Linear(T,70,2)(h)
    g = Graph(x, h)
    NN(g)
end
