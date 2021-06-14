# h = mot + doc au lieu de h = vcat(mot,doc)

using LinearAlgebra: length, Matrix
using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using Flux.Losses
using Flux: onehot, batch, onecold

using Plots

V = 64
Vdoc = 200

struct Dv2
    W::Matrix{Float64}
    D::Matrix{Float64}
    W1::Matrix{Float64}
end

Flux.@functor Dv2

function Dv2(vocsize::Int,docsize::Int,hidwordsize::Int)
    Dv2(rand(Float64,hidwordsize,vocsize).-0.5, rand(Float64,hidwordsize,docsize).-0.5, rand(Float64,vocsize,hidwordsize,).-0.5)
end

function (w::Dv2)(x,d)
    mot = w.W * x
    doc = w.D * d
    h = mot + doc
    y = w.W1 * h
    softmax(y)
end

data = dt2(1000)

typeof(data)
typeof(data[1])
size(data[1])
typeof(data[2])
size(data[2])
typeof(data[3])
size(data[3])

model2 = Dv2(V,Vdoc,10)

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

eval(model2(data[1],data[2]),data[3])
eval2(model2(data[1],data[2]),data[3])
eval3(model2(data[1],data[2]),data[3],s1,s2)

Ypred = model2(data[1],data[2])
Y = data[3]

j = 12

ypred = Ypred[:,j]
y = Y[:,j]
argmax(ypred)
onecold(y)
ypred[argmax(ypred)]


dat1 =  create_batch3("A01010101.txt",5)
dat2 = create_batch3("B07020101.txt",5)
dat3 =  create_batch3("A01010102.txt",5)

X = [dat1[1] dat2[1] dat3[1]]
d1 = [onehot(:1, [:1,:2,:3]) for i in 1:length(dat1[1][1,:]) ]
d2 = [onehot(:2, [:1,:2,:3]) for i in 1:length(dat2[1][1,:]) ]
d3 = [onehot(:3, [:1,:2,:3]) for i in 1:length(dat3[1][1,:]) ]

D = [batch(d1) batch(d2) batch(d3)]
Y = [dat1[2] dat2[2] dat3[2]]

data = (X,D,Y)

model2 = Dv2(V,3,20)
ps = params(model2)

loss = []

@time train!(cst2, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss2!(loss))

plot(loss)

eval(model2(data[1],data[2]),data[3])
eval2(model2(data[1],data[2]),data[3])
eval3(model2(data[1],data[2]),data[3],s1,s2)

data = (dat1[1], batch(d1), dat1[2])

model2 = Dv2(V,3,20)
ps = params(model2)

loss = []

@time train!(cst2, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss2!(loss))

plot(loss)

eval(model2(data[1],data[2]),data[3])
eval2(model2(data[1],data[2]),data[3])
eval3(model2(data[1],data[2]),data[3],s1,s2)

j = 5

ypred = Ypred[:,j]
y = Y[:,j]
argmax(ypred)
onecold(y)
ypred[argmax(ypred)]

#= g(x) = x-1

test = Chain( x -> x[1]^2+x[2]^2 , g )

test((2,3))

word(x) = Dense(V,hidword)
Doc(x) = Dense(Vdoc,hiddoc)
h(data) = vcat( word(data[1]) doc(data[2])
y(x) = Dense(hidword + hiddoc, V)

D2W = chain( data -> vcat( word(data[1]), doc(data[2]) ) , y   =#
