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
        fenetre = texte[inf:sup]
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
    vecteurs_mots = [
    ones(n)*10 .- (20).*(rand(n))
    for i in 1:64]
    vecteurs_contexte = [
    ones(n)*10 .- (20).*(rand(n))
    for i in 1:64]
    biais_mot = [10 - 20*rand() for i in 1:64]
    biais_contexte = [10 - 20*rand() for i in 1:64]
    return(vecteurs_mots, vecteurs_contexte, biais_mot, biais_contexte)
end

vecm_test, vecc_test, biaism_test, biaisc_test = wordvec(10)

f(x) = min( (x/10)^0.75 , 1)

function cost(vecteurs_mots,vecteurs_contexte,biais_mot, biais_contexte,mat)
    S = 0
    for i in 1:64

        wi = vecteurs_mots[i]
        bi = biais_mot[i]

        for j in 1:64
            m = mat[i][j]
            if m == 0
                break
            end
            
            wj = vecteurs_contexte[j]
            bj = biais_contexte[j]

            S += f(m)*( dot(wi,wj) + bi + bj - log(m) )^2
        end
    end
    S
end

cost(vecm_test, vecc_test, biaism_test, biaisc_test ,M2)

#= cost_test(x) = cost(vecm_test, vecc_test, biaism_test, biaisc_test, x)

cost_test(M2) =#

ps = params(vecm_test, vecc_test, biaism_test, biaisc_test)

#= ps2 = params(vecm_test[1], biaism_test, biaisc_stest)

ps3 = params(biaism_test, biaisc_test)

grad = gradient( () -> cost_test(M2) , ps )

grad2 = gradient( () -> cost_test(M2) , ps2 )

grad3 = gradient( () -> cost_test(M2) , ps3 )
 =#

#= grad[biaism_test]
grad3[biaism_test] =#
#= grad2[vecm_test[1]] =#

grad = gradient(ps) do 
    cost(vecm_test, vecc_test, biaism_test, biaisc_test, M2)
end

#= grad[vecm_test]
grad[vecc_test[1]]
v1 = vecc_test[1]
grad[v1]
grad[biaism_test] =#

opt = Descent()

#= a = [2.0]
b = [7.0]
c = [3.0]

f_test(a,b,c) = 2a[1] + b[1] + c[1]


gs = gradient(params(a,b,c)) do
         f_test(a,b,c)
    end

update!(opt, params(a,b,c), gs)

gs2 = gradient(params(a,b)) do
    f_test(a,b,c)
end

gs3 = gradient(params([a,b])) do
    f_test(a,b,c)
end =#


#= function opti(vecm, vecc, biaism, biaisc, M)

    gvm = [ zeros(10) for i in 1:64]
    gvc = [ zeros(10) for i in 1:64]

    gbm = [
        2 * sum([ f(M[i][j])*( dot(vecm[i],vecc[j]) + biaism[i] + biaisc[j] - log(M[i][j]) ) for j in 1:64])
        for i in 1:64]

    println(gbm)
    
        
    gbc = [
        2 * sum([ f(M[i][j])*( dot(vecm[i],vecc[j]) + biaism[i] + biaisc[j] - log(M[i][j]) ) for i in 1:64])
        for j in 1:64]

    
    for i in 1:64
        Mi = M[i]
        wi = vecm[i]
        bi = biaism[i]
        for k in 1:10
            gvm[i][k] = 2*sum([ f(Mi[j])* vecc[j][k] * ( dot(wi,vecc[j]) + bi + biaisc[j] - log(Mi[j]) ) for j in 1:64])
        end
    end

    for i in 1:64
        Mi = M[i]
        wi = vecc[i]
        bi = biaisc[i]
        for k in 1:10
            gvc[i][k] = 2*sum([ f(Mi[j])* vecm[j][k] * ( dot(wi,vecm[j]) + bi + biaisc[j] - log(Mi[j]) ) for j in 1:64])
        end
    end

    return(vecm - 0.1 * gvm , vecc - 0.1 * gvc , biaism - 0.1 * gbm , biaisc - 0.1 * gbc )
end =#


function opti(vecteurs_mots,vecteurs_contexte,biais_mot, biais_contexte,mat)

    gvm = [ zeros(10) for i in 1:64]
    gvc = [ zeros(10) for i in 1:64]

    gbm = zeros(64)
    gbc = zeros(64)

    
    for i in 1:64

        wi = vecteurs_mots[i]
        bi = biais_mot[i]

        for j in 1:64
            m = mat[i][j]
            if m == 0
                break
            end
            
            wj = vecteurs_contexte[j]                
                wj = vecteurs_contexte[j]
            wj = vecteurs_contexte[j]                
            bj = biais_contexte[j]

            gvm[i] += 2*f(m)*wj*( dot(wi,wj) + bi + bj - log(m) )
        end

    end

    for j in 1:64

        wj = vecteurs_contexte[j]
        bj = biais_contexte[j]

        for i in 1:64
            m = mat[j][i]
            if m == 0
                break
            end

            wi = vecteurs_mots[i]                
                wi = vecteurs_mots[i]
            wi = vecteurs_mots[i]                
            bi = biais_mot[i]
            
            gvc[j] += 2*f(m)*wi*( dot(wi,wj) + bi + bj - log(m) )
        end
    end

    for i in 1:64

        wi = vecteurs_mots[i]
        bi = biais_mot[i]

        for j in 1:64
            m = mat[i][j]
            if m == 0
                break
            end
            
            wj = vecteurs_contexte[j]            
            bj = biais_contexte[j]

            gbm[i] += 2*f(m)*( dot(wi,wj) + bi + bj - log(m) )
        end
    end

    for j in 1:64

        wj = vecteurs_contexte[j]
        bj = biais_contexte[j]

        for i in 1:64
            m = mat[i][j]
            if m == 0
                break
            end

            wi = vecteurs_mots[i]            
            bi = biais_mot[i]
            
            gbc[j] += 2*f(m)*( dot(wi,wj) + bi + bj - log(m) )
        end
    end


  return(gvm,gvc,gbm,gbc)
end

#= opti(vecm_test, vecc_test, biaism_test, biaisc_test, M2); =#

function maj2(vecteurs_mots,vecteurs_contexte,biais_mot, biais_contexte,mat)
    list = []
    for i in 1:100
        vecteurs_mots,vecteurs_contexte,biais_mot, biais_contexte = opti(vecteurs_mots,vecteurs_contexte,biais_mot, biais_contexte,mat)
        append!(list,[cost(vecteurs_mots,vecteurs_contexte,biais_mot, biais_contexte,mat)])
    end
    list
end

#= maj2(vecm_test, vecc_test, biaism_test, biaisc_test, M2,4)
gvm,gvc,gbm,gbc = opti(vecm_test, vecc_test, biaism_test, biaisc_test, M2) =#