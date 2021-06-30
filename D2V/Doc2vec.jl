using Flux 

struct Dv
    W::Matrix{Float64}
    D::Matrix{Float64}
    W1::Matrix{Float64}
end

Flux.@functor Dv

function Dv(vocsize::Int,docsize::Int,hidwordsize::Int,hiddocsize::Int)
    Dv(rand(Float64,hidwordsize,vocsize).-0.5,
    rand(Float64,hiddocsize,docsize).-0.5,
    rand(Float64,vocsize,hiddocsize+hidwordsize).-0.5)
end

function (w::Dv)(x,d)
    mot = w.W * x
    doc = w.D * d
    h = vcat(mot,doc)
    y = w.W1 * h
    softmax(y)
end

#= word(V,hidword) = Dense(V,hidword; bias = false)
doc(Vdoc,hiddoc) = Dense(Vdoc,hiddoc; bias = false)
h(V,hidword,hiddoc)(data) =  [word(V,hidword)(data[1]); doc(Vdoc,hiddoc)(data[2])]
y(V,hidword,hiddoc) = Dense(hidword + hiddoc, V; bias = false)

function D2V(V,hidword,hiddoc)
    return Chain( h(V,hidword,hiddoc) , y(V,hidword,hiddoc) , softmax)
end =#

function update_loss!(list,loss)
    push!(list, loss(data...))
end