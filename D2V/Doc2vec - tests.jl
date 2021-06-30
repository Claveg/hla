using Plots
using Flux
using SparseArrays

include("W2V\\Donnees.jl")
include("D2V\\Doc2vec.jl")
include("D2V\\D2V_donnees.jl")
include("D2V\\imgt.jl")

V = 64
Vdoc = n
hiddoc = 100
hidword = 10
c = 5

model = Dv(V,Vdoc,hidword,hiddoc) |> gpu
ps = Flux.params(model)

cost(x_w,x_d,y) = Flux.crossentropy( model(x_w,x_d), y)

lost = []

opt = ADAGrad()

epoch = 1
batchsize = 10

db2 = copy(db1[1:100])

function entraine(epoch,batchsize)
    for e in 1:epoch
        for  i in 1:batchsize:100
            data = imgt(db2[i:i+batchsize-1],c)
            gpu(data)
            @time Flux.train!(cost, ps, [(X,D,Y)], opt)
        end
    end
end

@time entraine(epoch,batchsize)

plot(lost)

(lost[1]-lost[end])/lost[1]

model2 = Dv(V,Vdoc,hidword,hiddoc)
ps2 = Flux.params(model2)
opt = ADAGrad()

lost = []

@time train!(cst2, ps2, Iterators.repeated(data,19), opt; cb = () -> update_loss2!(lost))


# avec des Dense

word = Dense(V,hidword; bias = false)
doc = Dense(Vdoc,hiddoc; bias = false)
h(data) =  [word(data[1]); doc(data[2])]
y = Dense(hidword + hiddoc, V; bias = false)

D2V = Chain( h , y , softmax)

function loss(x,d,y)
    Flux.crossentropy( D2V((x,d)), y)
end

@time loss(X,D,Y)

@time (X,D,Y) = imgt(db2[1:5],5)

X = Array(X)

epoch = 1
batchsize = 10

db2 = copy(db1[1:100])

ps = params(D2V)
opt = ADAGrad()

function entraine(epoch,batchsize)
    for e in 1:epoch
        for  i in 1:batchsize:100
            data = imgt(db2[i:i+batchsize-1],c)
            @time Flux.train!(loss, ps, [(X,D,Y)], opt)
        end
    end
end

data = imgt(db2[1:5],5)

@time entraine(epoch,batchsize)

#= X = gpu(X)
Y = gpu(Y)
D = gpu(D)

@time cost(X,D,Y) =#
