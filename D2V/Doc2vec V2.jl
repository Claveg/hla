# tests avec des données "aléatoires" et de vraies allèles

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

model2 = Dv(V,3,30,3)

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
ypred[43]


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

model2 = Dv(V,3,10,20)
ps = params(model2)

loss = []

@time train!(cst2, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss2!(loss))

plot(loss)

eval(model2(data[1],data[2]),data[3])
eval2(model2(data[1],data[2]),data[3])
eval3(model2(data[1],data[2]),data[3],s1,s2)

# essais avec Dense

g(x) = x-1

test = Chain( x -> x[1]^2+x[2]^2 , g )

test((2,3))

hidword = 20
hiddoc = 20
Vdoc = 3

word = Dense(V,hidword; bias = false)
doc = Dense(Vdoc,hiddoc; bias = false)
y = Dense(hidword + hiddoc, V; bias = false)

D2W = Chain( data -> vcat( word(data[1]), doc(data[2]) ) , y , softmax)

D2W( [ rand(V) , rand(Vdoc) ] )

ps = params(D2W)

function cst3(x_w,x_d,y)
    crossentropy( D2W([x_w,x_d]),y ; agg = sum)
end

#= Flux.@functor D2W =#

cst3(data...)

Wtest = word.W[:,:]
Dtest = doc.W[:,:]
W1test = y.W[:,:]

loss = []

function update_loss3!(ls)
    push!(ls, cst3(data...))
end

opt = ADAGrad()

@time train!(cst3, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss3!(loss))

plot(loss)

Wtest - word.W[:,:]
Dtest - doc.W[:,:]
W1test - y.W[:,:]

eval(D2W((data[1],data[2])),data[3])
eval2(D2W((data[1],data[2])),data[3])
eval3(D2W((data[1],data[2])),data[3],s1,s2)

#### version 2 

word2 = Dense(V,hidword; bias = false)
doc2 = Dense(Vdoc,hiddoc; bias = false)
h2(data) =  [word2(data[1]); doc2(data[2])]
y2 = Dense(hidword + hiddoc, V; bias = false)

D2W2 = Chain( h2 , y2 , softmax)

ps2 = params(D2W2)


word2( rand(V) )
doc2( rand(Vdoc) )
h2( [rand(V) , rand(Vdoc) ])
y2( rand(V + Vdoc) )

y2( h2( ( rand(V)  ,  rand(Vdoc) ) ))

D2W2( (rand(V) , rand(Vdoc) ) )

function cst4(x_w,x_d,y)
    crossentropy( D2W2([x_w,x_d]),y ; agg = sum)
end

ps = params(D2W2)
cst4(data...)

Wtest = word2.W[:,:]
Dtest = doc2.W[:,:]
W1test = y2.W[:,:]

loss = []

function update_loss4!(ls)
    push!(ls, cst4(data...))
end

opt = ADAGrad()

@time train!(cst4, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss4!(loss))

plot(loss)

Wtest - word.W[:,:]
Dtest - doc.W[:,:]
W1test - y.W[:,:]