using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using Plots

bases = "ATGC"
l = Set([])
for i in 1:4
    for j in 1:4
        for k in 1:4
            push!(l,bases[i]*bases[j]*bases[k])
        end
    end
end

vocabulaire = collect(l)
sort!(vocabulaire)

function create_dic()
    Dict([codon => Dict([cod => 0 for cod in vocabulaire])
    for codon in vocabulaire])
end

function co_oc!(dic,texte,window)
    n = length(texte)
    for i in 1:n
        mot = texte[i]
        inf = max(1,i-window)
        sup = min(n, i+window)
        fenetre = texte[inf:sup] # oups
        for codon in fenetre
            dic[mot][codon]+=1 
        end
    end
    return dic
end


function allele(n)
    a = [bases[rand(1:4)]*bases[rand(1:4)]*bases[rand(1:4)] for i in 1:n]
end

test = allele(1000)
dico_test = create_dic()

co_oc!(dico_test,test,5)

function matrice(dic)
    m = [
        [dic[mot][word] for word in vocabulaire]
        for mot in vocabulaire]    
end

M2 = matrice2(dico_test);

struct Glove
    W_main::Matrix{Float64}
    W_ctx::Matrix{Float64}
    b_main::Vector{Float64}
    b_ctx::Vector{Float64}
end

function Glove(vocabsize::Int, vecsize::Int)
    shift = Float64(0.5)
    Glove(
        (rand(Float64, vecsize, vocabsize) .- shift) ./ Float64(vecsize + 1),
        (rand(Float64, vecsize, vocabsize) .- shift) ./ Float64(vecsize + 1),
        (rand(Float64, vocabsize) .- shift) ./ Float64(vecsize + 1),
        (rand(Float64, vocabsize) .- shift) ./ Float64(vecsize + 1) )
end

g = Glove(64,10)

f(x) = min( (x/10)^0.75 , 1)

function cost2(mat::Vector{Vector{Int64}})

    S = 0
    for i in 1:64

        wi = g.W_main[:,i]
        bi = g.b_main[i]

        for j in 1:64
            m = mat[i][j]
            if m == 0
                break
            end
            
            wj = g.W_ctx[:,j]
            bj = g.b_ctx[j]

            S += f(m)*( dot(wi,wj) + bi + bj - log(m) )^2
        end
    end
    S
end

Flux.@functor Glove

cost2(M2)

function update_loss!()
    push!(train_loss, cost2(M2))
    println("train loss = $(train_loss[end])")
end

g = Glove(64,10)
train_loss = [cost2(M2)]
p = params(g)
opt = ADAGrad()
Flux.train!(cost2, p, Iterators.repeated(M2, 100), opt; cb = update_loss!)

plot(1:length(train_loss), train_loss, xlabel="epochs", ylabel="loss")

@time Flux.train!(cost2, p, Iterators.repeated(M2, 100), opt)