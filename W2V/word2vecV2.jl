using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using Flux.Losses
using Flux: onehot, batch

V = 64

struct W2v
    W::Matrix{Float64}
    W1::Matrix{Float64}
end

Flux.@functor W2v

function W2v(vocsize::Int,hidsize::Int)
    W2v(rand(Float64,hidsize,vocsize).-0.5, rand(Float64,hidsize,vocsize,).-0.5)
end

function concat(X)
    n = length(X[1,:])
    sum(X[:,i] for i in 1:n)/n
end

function (w::W2v)(x,y)
    h = w.W * concat(x)
    v_main = w.W1 * y
    score = exp( dot(v_main,h) )
    S = sum( exp( dot(w.W1[:,i],h) ) for i in 1:V )
    return(score/S)
end

function cst3(x,y)
    -log(model3(x,y))
end

labels = collect(1:v)

function dt(corpussize::Int)
    data = []
    for i in 1:corpussize

        rd = [ onehot(rand(1:V),labels) for j in 1:5]

        x = batch(rd)
        y = onehot(rand(1:V),labels)

        append!(data,[(x,y)])
    end
    data
end

data = dt(1000)

model3 = W2v(64,10)

W = model3.W[:,:]
W1 = model3.W1[:,:]

ps = params(model3)

opt = Descent()

cst3(data[1][1],data[1][2])
cst3(data[1]...)

function temp(dat)
    S = 0
    for d in dat
        S+= cst3(d...)
    end
    S
end

temp(data)

@time train!(cst3, ps, [data...], opt)

function entraine(n)
    train_loss = zeros(n)
    for i in 1:n
        train!(cst3, ps, [data...], opt)
        train_loss[i] = temp(data)
    end
    train_loss
end

plot(entraine(50))

dW = model3.W-W
dW1 = model3.W1-W1

