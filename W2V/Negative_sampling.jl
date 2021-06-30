using LinearAlgebra: Matrix

using StatsBase
using Flux
using Flux: onecold, onehot, onehotbatch
using LinearAlgebra

k = 3

#sampling output word

function p_n(out::Int)
    wv = ProbabilityWeights(ones(V)/(V-1))
    wv[out] = 0
    samples = sample(1:64,wv,k; replace = false)
    return samples
end

#sampling context word

#= function p_n(out::Vector{Int})
    c = lenght(out)
    wv = ProbabilityWeights(ones(V)/V)
    samples = [sample(1:64,wv,c; replace = false) for i in 1:k]
    return samples # if out in samples ??
end =#

struct WvN
    W1::Matrix{Float64}
    W2::Matrix{Float64}
end

Flux.@functor WvN

function WvN(vocsize::Int,hidsize::Int)
    WvN( (rand(Float64,hidsize,vocsize).-0.5) / (vocsize+1),
         rand(Float64,hidsize,vocsize).-0.5 / (vocsize+1) )
end

ls(x::Vector{Float64}) = - log.( σ.( x ) ) 
ls(x::Float64) = - log( σ( x ) )
ls(x::Matrix{Float64}) = - log.( σ.( x ) ) 

function loss(x,y)

    h = w.W1*x
    v = w.W2*y
    E = ls( dot(v,h) )

    out = onecold(y,collect(1:V))
    samples = p_n(out)
    
    for s in samples
        
        ns = w.W2[:,s]
        E += ls( - dot(ns,h) )

    end    

    return sum(E)
end

w = WvN(V,10)

x1 = rand(V)
x2 = rand(V)

@time loss(x1,x2)

#= oh = Flux.onehot(32,1:64)
W = rand(10,64)

@time h = W*oh

@time h = copy(W[:,32])

S=0
for i in 1:10000
    S+= @elapsed W*oh
end
S =#

function pn(out::Int)
    wv = ProbabilityWeights(ones(V)/(V-1))
    wv[out] = 0
    s = sample(1:64,wv)
    return s
end

function pn_mat(Y::Flux.OneHotArray)

    sampl = onecold(Y,1:64)
    sampl = pn.(sampl)

    S = onehotbatch(sampl,1:64)

    return S    
end

function cout(x,y)

    h = w.W1*x
    v = w.W2*y
    E = ls( sum(h.*v, dims=1) )

    for i in 1:k

        S = pn_mat(y)
        ns = w.W2*S

        E += ls( -sum(ns.*h, dims=1) )

    end

    return sum(E)
end

N = 5000
x = rand(V,N)
y = onehotbatch(rand(1:V,N),1:V)

@time cout(x,y)

data = create_batch("D:\\Mines\\Stage IRSL\\hla\\Sequences\\A01010101.txt",5)

ps = params(w)
opt = ADAGrad()

@time Flux.train!(cout, ps, [data], opt)





