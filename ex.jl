using Random
using LinearAlgebra
using Flux

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
    Dict([codon => Dict([codon => 0 for codon in vocabulaire])
    for codon in vocabulaire])
end

dico["AAA"]["ATG"]

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
    [
    ([ones(n)*10 .- (20).*(rand(n))],10 - 20*rand())
    for i in 1:64]
end

word_test = wordvec(10)

f(x) = min( (x/20)^0.75 , 1)

function cost(wordvec,mat)
    S = Float32(0)
    for i in 1:2
        for j in 1:64
            m = mat[i][j]
            if m == 0
                break
            end
            wi = wordvec[i]
            wj = wordvec[j]
            S += f(m)*( dot(wi[1],wj[1]) + wi[2] + wj[2] - log(m) )^2
        end
    end
    S
end

cost(word_test,M2)

p = params(word_test)

#= grad = gradient(x -> cost(word_test,x) , p) =#