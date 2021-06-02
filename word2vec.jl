using LinearAlgebra: length
using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using Flux.Losses
using Flux: onehot, onecold, batch

using Plots

V = 64

struct Wv
    W::Matrix{Float64}
    W1::Matrix{Float64}
end

Flux.@functor Wv

function Wv(vocsize::Int,hidsize::Int)
    Wv(rand(Float64,hidsize,vocsize).-0.5, rand(Float64,vocsize,hidsize,).-0.5)
end

function concat(X)
    n = length(X[1,:])
    sum(X[:,i] for i in 1:n)
end

(w::Wv)(x) = softmax( w.W1 * (w.W * concat(x)) )

function cst(x,y)
    crossentropy(model(x),y)
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

model = Wv(64,10)

W = model.W[:,:]
W1 = model.W1[:,:]

ps = params(model)

opt = Descent()

cst(data[1][1],data[1][2])
cst(data[1]...)

function temp(dat)
    S = 0
    for d in dat
        S+= cst(d...)
    end
    S
end

temp(data)

@time train!(cst, ps, [data...], opt)

function entraine(n)
    train_loss = zeros(n)
    for i in 1:n
        train!(cst, ps, [data...], opt)
        train_loss[i] = temp(data)
    end
    train_loss
end

plot(entraine(50))

dW = model.W-W
dW1 = model.W1-W1

#= train!(cst, ps, Iterators.repeated([data...], 2), opt) =#

#= a = [0,1,0]
b = onehot(1,labels)
c = onehot(3, labels)

test = [b c]

h = model.W*concat(test)

y = model.W1*h

z = softmax(y)

model(test)

cst(test,b)

concat(test)

M1 = [[1,2,3] [4,5,6]]
M2 = [[1,2] [3,4] [5,6]] =#
