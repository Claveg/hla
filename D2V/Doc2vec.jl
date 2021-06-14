using LinearAlgebra: length, Matrix
using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using Flux.Losses
using Flux: onehot, onecold, batch

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

function concat(X)
    n = length(X[1,:])
    sum(X[:,i] for i in 1:n)
end

function (w::Dv)(m,d)
    mot = w.W * concat(m)
    doc = w.D * d
    h = vcat(mot,doc)
    y = w.W1 * h
    softmax(y)
end

function cst2(x,y) #Tuple{Matrix{Float64},Vector{Float64}}
    (m,d) = x
    crossentropy(model2(m,d),y)
end

labels = collect(1:V)
labelsdoc = collect(1:Vdoc)

function dt2(corpussize::Int)
    data = []
    for i in 1:corpussize

        rd = [ onehot(rand(1:V),labels) for j in 1:5]

        x1 = batch(rd)
        x2 = onehot(rand(1:Vdoc),labelsdoc)
        x = (x1,x2)

        y = onehot(rand(1:V),labels)

        append!(data,[(x,y)])
    end
    data
end

data = dt2(1000)

model2 = Dv(V,Vdoc,10,10)

W = model2.W[:,:]
D = model2.D[:,:]
W1 = model2.W1[:,:]

ps = params(model2)

opt = Descent()

cst2(data[1]...)

function temp2(dat)
    S = 0
    for d in dat
        S+= cst2(d...)
    end
    S
end

temp2(data)

@time train!(cst2, ps, [data...], opt)

function entraine2(n)
    train_loss = zeros(n)
    for i in 1:n
        train!(cst2, ps, [data...], opt)
        train_loss[i] = temp2(data)
    end
    train_loss
end

plot(entraine2(50))

dW = model2.W-W
dD = model2.D-D
dW1 = model2.W1-W1