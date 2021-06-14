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
    vecteurs_main = [
    ones(n) .- (2).*(rand(n))
    for i in 1:64]
    vecteurs_contexte = [
    ones(n) .- (2).*(rand(n))
    for i in 1:64]
    biais_main = [1 - 2*rand() for i in 1:64]
    biais_contexte = [1 - 2*rand() for i in 1:64]
    return(vecteurs_main, vecteurs_contexte, biais_main, biais_contexte)
end

vecm_test, vecc_test, biaism_test, biaisc_test = wordvec(10);

f(x) = min( (x/10)^0.75 , 1)

function cost(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte,mat)
    S = 0
    for i in 1:64

        wi = vecteurs_main[i]
        bi = biais_main[i]

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

ps = params(vecm_test, vecc_test, biaism_test, biaisc_test);

#= ps2 = params(vecm_test[1], biaism_test, biaisc_stest)

ps3 = params(biaism_test, biaisc_test) =#

grad = gradient(ps) do 
    cost(vecm_test, vecc_test, biaism_test, biaisc_test, M2)
end

grad[biaism_test]

#= grad[vecm_test]
grad[vecc_test[1]]
v1 = vecc_test[1]
grad[v1]
grad[biaism_test] =#

opt = Descent(0.01)

function maj(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte,mat)

    list = [cost(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte,mat)]

    ps = params(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte)

    g = gradient(ps) do 
        cost(vecm_test, vecc_test, biaism_test, biaisc_test, M2)
    end

    for i in 1:100
        update!(opt, ps, g)
        append!(list,[cost(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte,mat)])

        ps = params(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte)

        g = gradient(ps) do 
            cost(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte, mat)
        end
    end
    return(list)
end

l = maj(vecm_test, vecc_test, biaism_test, biaisc_test, M2)


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
end 

gs3[a]
gs3[[2.0]]=#



function opti(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte,mat)

    gvm = [ zeros(10) for i in 1:64]
    gvc = [ zeros(10) for i in 1:64]

    gbm = zeros(64)
    gbc = zeros(64)

    
    for i in 1:64

        wi = vecteurs_main[i]
        bi = biais_main[i]

        for j in 1:64
            m = mat[i][j]
            if m == 0
                break
            end
            
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

            wi = vecteurs_main[i]                
            bi = biais_main[i]
            
            gvc[j] += 2*f(m)*wi*( dot(wi,wj) + bi + bj - log(m) )
        end
    end

    for i in 1:64

        wi = vecteurs_main[i]
        bi = biais_main[i]

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

            wi = vecteurs_main[i]            
            bi = biais_main[i]
            
            gbc[j] += 2*f(m)*( dot(wi,wj) + bi + bj - log(m) )
        end
    end


  return(gvm,gvc,gbm,gbc)
end

#= opti(vecm_test, vecc_test, biaism_test, biaisc_test, M2); =#

function maj2(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte,mat)
    list = [cost(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte,mat)]
    η = 0.01
    for i in 1:100
        gvm, gvc ,gbm, gbc = opti(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte,mat)
        vecteurs_main -= η*gvm
        vecteurs_contexte -= η*gvc
        biais_main -= η*gbm
        biais_contexte -= η*gbc
        append!(list,[cost(vecteurs_main,vecteurs_contexte,biais_main, biais_contexte,mat)])
    end
    return(list,vecteurs_main,vecteurs_contexte,biais_main, biais_contexte)
end

#= list, vm, vc, bm, bc = maj2(vecm_test, vecc_test, biaism_test, biaisc_test, M2);
gvm,gvc,gbm,gbc = opti(vecm_test, vecc_test, biaism_test, biaisc_test, M2) =#

