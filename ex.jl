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
    for i in window+1:n-window
        mot = texte[i]
        fenetre = texte[i-window:i+window]
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
    collect([collect(values(d)) for d in values(dic)])
end

function matrice2(dic)
    m = [
        [dic[mot][word] for word in vocabulaire]
        for mot in vocabulaire]    
end

@time M = matrice(dico_test);

@time M2 = matrice2(dico_test);

function wordvec(n)
    vecteurs = [
    [ones(n)*10 .- (20).*(rand(n))]
    for i in 1:64]
    biais = [10 - 20*rand() for i in 1:64]
    return(vecteurs, biais)
end

vec_test, biais_test = wordvec(10)

f(x) = min( (x/20)^0.75 , 1)

function cost(vecteurs,biais,mat)
    S = Float32(0)
    for i in 1:64
        for j in 1:64
            m = mat[i][j]
            if m == 0
                break
            end
            wi = vecteurs[i]
            wj = vecteurs[j]
            bi = biais[i]
            bj = biais[j]
            S += f(m)*( dot(wi,wj) + bi + bj - log(m) )^2
        end
    end
    S
end

cost(vec_test,biais_test,M2)

cost_test(x) = cost(vec_test, biais_test, x)

cost_test(M2)

ps = params(vec_test, biais_test)

grad = gradient( () -> cost_test(M2) , ps )

grad[biais_test]
grad[vec_test[1]]

opt = Descent()

function maj()
    list = []
    for i in 1:100
        update!(opt, ps, grad)
        append!(list,[cost_test(M2)])
    end
    list
end

#= list = maj()

plot(list) =#
