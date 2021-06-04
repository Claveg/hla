using LinearAlgebra: length, Matrix
using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using Flux.Losses
using Flux: onehot, batch

using Plots

V = 64
Vdoc = 200

struct Dv
    W::Matrix{Float64}
    D::Matrix{Float64}
    W1::Matrix{Float64}
end

Flux.@functor Dv

function Dv(vocsize::Int,docsize::Int,hidwordsize::Int,hiddocsize::Int)
    Dv(rand(Float64,hidwordsize,vocsize).-0.5, rand(Float64,hiddocsize,docsize).-0.5, rand(Float64,vocsize,hiddocsize+hidwordsize,).-0.5)
end


function (w::Dv)(x,d)
    mot = w.W * x
    doc = w.D * d
    h = vcat(mot,doc)
    y = w.W1 * h
    softmax(y)
end

function cst2(w,d,y)
    crossentropy(model2(w,d),y; agg = sum)
end

labels = collect(1:V)
labelsdoc = collect(1:Vdoc)

function ht(v::Int,n::Int)
    l = []

    for i in 1:n
        vec = zeros(v)
        rd = sample(1:V, C, replace = false)
        for r in rd
            vec[r] = 1
        end
        push!(l,vec)
    end
    batch(l)    
end

function dt2(corpusSize::Int)

    X = ht(V,corpusSize)
    Doc = batch( [onehot(rand(1:Vdoc),labelsdoc) for i in 1:corpusSize] )
    Y = batch( [onehot(rand(1:V),labels) for i in 1:corpusSize] )

    return (X,Doc,Y)
end

function update_loss2!(ls)
    push!(ls, cst2(data...))
end

data = dt2(1000)

typeof(data)
typeof(data[1])
size(data[1])
typeof(data[2])
size(data[2])
typeof(data[3])
size(data[3])

model2 = Dv(V,Vdoc,10,10)

W = model2.W[:,:]
D = model2.D[:,:]
W1 = model2.W1[:,:]

ps = params(model2)

opt = ADAGrad()

cst2(data...)

@time train!(cst2, ps, [data], opt)

loss = []

@time train!(cst2, ps, Iterators.repeated(data,500), opt; cb = () -> update_loss2!(loss))

plot(loss)





dW = model2.W-W
dD = model2.D-D
dW1 = model2.W1-W1