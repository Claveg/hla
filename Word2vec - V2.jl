using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using Flux.Losses
using Flux: onehot, batch
using StatsBase: sample, mean

using Plots

V = 64
C = 10

struct Wv
    W::Matrix{Float64}
    W1::Matrix{Float64}
end

Flux.@functor Wv

function Wv(vocsize::Int,hidsize::Int)
    Wv(rand(Float64,hidsize,vocsize).-0.5, rand(Float64,vocsize,hidsize,).-0.5)
end

(w::Wv)(x) = softmax( w.W1 * (w.W * x) ) # z: vector Vx1 (ou matrice de z VxcorpusSize)

# m = Wv(10,5)
# xtest3 = ones(50,100)
# m(xtest3)

function cst(x,y)
    crossentropy(model(x),y;agg = sum)
end

# @time dot( model(data[1][1]), data[1][2] )
# @time model(data[1][1])[6]

labels = collect(1:V)

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

function dt(corpusSize::Int)

    X = ht(V,corpusSize)
    Y = batch( [onehot(rand(1:V),labels) for i in 1:corpusSize] )

    return (X,Y)

end

data = dt(1000)

model = Wv(V,10)

model(data[1][:,1])
model(data[1])

W = model.W[:,:]
W1 = model.W1[:,:]

ps = params(model)

opt1 = Descent(.01)
opt2 = Descent(.001)
opt3 = ADAGrad()
opt4 = ADAM()

cst(data[1][:,1],data[2][:,1])
cst(data...)

@time train!(cst, ps, [data], opt)

loss = []

function update_loss!(ls)
    push!(ls, cst(data...))
end

@time train!(cst, ps, Iterators.repeated(data,1000), opt; cb = () -> update_loss!(loss))

dW = model.W-W
dW1 = model.W1-W1

@time plot(loss)

loss1 = []
model = Wv(V,10)
ps = params(model)
@time train!(cst, ps, Iterators.repeated(data,1000), opt1; cb = () -> update_loss!(loss1))

loss2 = []
model = Wv(V,10)
ps = params(model)
@time train!(cst, ps, Iterators.repeated(data,1000), opt2; cb = () ->update_loss!(loss2))

loss3 = []
model = Wv(V,10)
ps = params(model)
@time train!(cst, ps, Iterators.repeated(data,1000), opt3; cb = () ->update_loss!(loss3))

loss4 = []
model = Wv(V,10)
ps = params(model)
@time train!(cst, ps, Iterators.repeated(data,1000), opt4; cb = () ->update_loss!(loss4))

img = plot(1:length(loss1),loss1, label = "SGD 0.01")
plot!(1:length(loss2),loss2, label = "SGD 0.001")
plot!(1:length(loss3),loss3, label = "ADAGrad")
plot!(1:length(loss4),loss4, label = "ADAM")

savefig(img,"optW2V.png")


#= train!(cst, ps, Iterators.repeated([data...], 2), opt) =#

#= a = [0,1,0]
b = onehot(1,labels)
c = onehot(3, labels)

test = [b c]

h = model.W*concat4(test)

y = model.W1*h

z = softmax(y)

model(test)

cst(test,b)

concat4(test)

M1 = [[1,2,3] [4,5,6]]
M2 = [[1,2] [3,4] [5,6]] =#

# function concat3(X)

#     n,p = (length(X[:,1]) , length(X[1,:]) )
#     q = Int(n/5)
#     M = zeros(q,p)

#     for j in 1:p
#         c = X[:,j]
#         c = reshape(c, q, 5)
#         M[:,j] = sum(c[:,i] for i in 1:5)
#     end
#     M    
# end

# function concat4(X)

#     n,p = (length(X[:,1]) , length(X[1,:]) )
#     q = Int(n/5)
#     M = zeros(q,p)

#     for j in 1:p
#         c = X[:,j]
#         for i in 1:q
#             M[i,j] = sum( c[i+k] for k in (0:4)*q )
#         end
#     end
#     M    
# end

# xtest = ones(10,5)
# xtest2 = ones(10)
# @time concat3(xtest)
# @time concat4(xtest)

# concat3(xtest2)
# concat4(xtest2)

#= typeof(data)
typeof(data[1])
size(data[1])
typeof(data[2])
size(data[2]) =#
