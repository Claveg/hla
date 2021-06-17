using Flux
using Random

struct Wv
    W::Matrix{Float64}
    W1::Matrix{Float64}
end

Flux.@functor Wv

function Wv(vocsize::Int,hidsize::Int)
    Wv( (rand(Float64,hidsize,vocsize).-0.5) / (vocsize+1),
         rand(Float64,vocsize,hidsize,).-0.5 / (vocsize+1) )
end

(w::Wv)(x) = softmax( w.W1 * (w.W * x) )

h(V,hidword) = Dense(V,hidword; bias = false)
y(V,hidword) = Dense(hidword, V; bias = false)

function W2V(V,hidword)
    return Chain( h(V,hidword) , y(V,hidword) , softmax)
end

function update_loss!(list,loss)
    push!(list, loss(data...))
end