using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using Plots
using Statistics

xmax = 20
V = 64

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
    Dict([codon => Dict([cod => 0.0 for cod in vocabulaire])
    for codon in vocabulaire])
end

function co_oc!(dic,texte,window; poids = x->1/x)
    n = length(texte)
    for i in 1:n
        mot = texte[i]
        inf = max(1,i-window)
        sup = min(n, i+window)

        for j in inf:sup
            if j!=i 
                dic[mot][texte[j]]+=poids(abs(j-i))
            end
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

M2 = matrice(dico_test)

struct Glove
    W_main::Matrix{Float64}
    W_ctx::Matrix{Float64}
    b_main::Vector{Float64}
    b_ctx::Vector{Float64}
end

function Glove(vocabsize::Int, vecsize::Int)
    Glove(
        ones(Float64, vecsize, vocabsize),
        ones(Float64, vecsize, vocabsize),
        ones(Float64, vocabsize),
        ones(Float64, vocabsize))
end

Flux.@functor Glove

g = Glove(64,10)

f(x) = min( (x/xmax)^0.75 , 1)

function cost2(mat::Vector{Vector{Float64}})

    S = 0
    for i in 1:V

        wi = g.W_main[:,i]
        bi = g.b_main[i]

        for j in 1:V
            m = mat[i][j]
            if m != 0
                wj = g.W_ctx[:,j]
                bj = g.b_ctx[j]

                S += f(m)*( dot(wi,wj) + bi + bj - log(m) )^2
            end
        end
    end
    S
end

function cost(mat::Vector{Vector{Float64}})

    S = 0
    vs = length(g.W_main[:,1])
    for i in 1:V

        wi = g.W_main[:,i]
        bi = g.b_main[i]

        for j in 1:V
            m = mat[i][j]
            if m != 0
            
                wj = g.W_ctx[:,j]
                bj = g.b_ctx[j]
                ps = 0

                for k in 1:vs
                    ps += wi[k]*wj[k]
                end

                S += f(m)*( ps + bi + bj - log(m) )^2
            end
        end
    end
    S
end

cost2(M2)

function update_loss!(mat)
    push!(train_loss, cost2(mat))
    println("train loss = $(train_loss[end])")
end

# lire les sequences

function clean(sequence)
    replace(sequence, r"[\n\r\s]"=>"")
end

function lire(name)
    f = open(name) do file
        read(file, String)
    end
    clean(f)
end

A1 = lire("A01010101.txt")
A1bis = lire("A01010101bis.txt")
Atest = lire("seqtest.txt")
A2 = lire("A01010102.txt")
B = lire("B07020101.txt")


function vecSequence(sequence)
    l = []
    n = length(sequence)
    i = 1
    while i+2<n
        append!(l,[sequence[i:i+2]])
        i+=3
    end
    l
end

dico_test = create_dic()
co_oc!(dico_test,vecSequence(Atest),3; poids = x->4-x)
Mtest = matrice(dico_test)

dico_A1 = create_dic()
dico_A1bis = create_dic()
dico_A2 = create_dic()
dico_B = create_dic()

co_oc!(dico_A1,vecSequence(A1),5; poids = x->6-x)
co_oc!(dico_A1bis,vecSequence(A1bis),5; poids = x->6-x)
co_oc!(dico_A2,vecSequence(A2),5; poids = x->6-x)
co_oc!(dico_B,vecSequence(B),5; poids = x->6-x)

MA1 = matrice(dico_A1)
MA1bis = matrice(dico_A1bis)
MA2 = matrice(dico_A2)
MB = matrice(dico_B)

norm(MA1-MA1bis)
norm(MA1-MA2)
norm(MA1-MB)

#= MA1 = [zeros(Float64,64) for i in 1:64] =#
MA1 = [rand(Float64,64).*1000 for i in 1:64]
g = Glove(64,10)
p = params(g)
cost(MA1)
WA1_maino = copy(g.W_main)
WA1_ctxo = copy(g.W_ctx)

(1/20)^0.75*(10+2-log(1))^2

opt = Flux.Optimise.ADAGrad()
Flux.train!(cost, p, Iterators.repeated(MA1, 50), opt)
cost(MA1)

WA1_main = g.W_main
WA1_ctx = g.W_ctx

WA1_main-WA1_maino

g = Glove(64,10)
p = params(g)
opt = Flux.Optimise.ADAGrad()
Flux.train!(cost2, p, Iterators.repeated(MA1, 50), opt)

WA1_main2 = g.W_main
WA1_ctx2 = g.W_ctx

g = Glove(64,10)
p = params(g)
Flux.train!(cost2, p, Iterators.repeated(MA2, 50), opt)

WA2_main = g.W_main
WA2_ctx = g.W_ctx

g = Glove(64,10)
p = params(g)
Flux.train!(cost2, p, Iterators.repeated(MB, 50), opt)

WB_main = g.W_main
WB_ctx = g.W_ctx

#PCA et plots

using MultivariateStats

pcA1 = MultivariateStats.fit(PCA, WA1_main+WA1_ctx; maxoutdim=2)
Y = transform(pcA1, WA1_main+WA1_ctx)

pcA2 = MultivariateStats.fit(PCA, WA2_main+WA2_ctx; maxoutdim=2)
Y2 = transform(pcA2, WA2_main+WA2_ctx)

pcB = MultivariateStats.fit(PCA, WB_main+WB_ctx; maxoutdim=2)
Y3 = transform(pcB, WB_main+WB_ctx)


plotly()

scatter(Y[1,:],Y[2,:], hover = vocabulaire, hovermode="closest")
scatter(Y2[1,:],Y2[2,:], hover = vocabulaire)
scatter(Y3[1,:],Y3[2,:], hover = vocabulaire,
series_annotations = [text("   "*voc, Plots.font("Sans", 8), :left) 
 for voc in vocabulaire])

#= scatter(Y[1,:],Y[2,:], hover = vocabulaire,
 series_annotations = [text("   "*voc, Plots.font("Sans", 8), :left) 
 for voc in vocabulaire]) =#

# essais pour comparer les embeddings des sequences 

 function taille(v,i)
    n = length(v[:,1])
    S = sum([norm(v[:,j]) for j in 1:n])
    return(norm(v[:,i])/S)
 end

 WA1 = WA1_main+WA1_ctx;
 WA2 = WA2_ctx+WA2_main;
 WA12 = WA1_main2+WA1_ctx2;
 WA1bis = WA1_mainb+WA1_ctxb;
 WB = WB_main+WB_main;

 norm(WA1-WA2)
 norm(WA1-WA12)
 norm(WA1-WB)
 norm(WA1-WA1bis)
 

m1 = mean([norm(WA1[:,i]-WA2[:,i]) for i in 1:V])
m2 = mean([norm(WA1[:,i]-WB[:,i]) for i in 1:V])

m1 = mean(x->x^2,[norm(WA1[:,i])-norm(WA2[:,i]) for i in 1:V])
m2 = mean(x->x^2,[norm(WA1[:,i])-norm(WB[:,i]) for i in 1:V])
m3 = mean(x->x^2,[norm(WA1[:,i])-norm(WA12[:,i]) for i in 1:V])

function s(v1,v2)
    S = 0.0
    for i in 1:V-1
        for j in i+1:V
            S += ( norm(v1[:,i]-v1[:,j])-norm(v2[:,i]-v2[:,j]) )^2
        end
    end
    S
end

s(WA1,WA2)
s(WA1,WA1bis)
s(WA1,WA1)
s(WA1,WB)
s(WA2,WB)
s(WA1,WA12)

function similaire(W,i::Int64,r)
    l = [ (-1.0, 0) for i in 1:V ]
    # d = Dict{AbstractFloat,Int8}()
    
    for j in 1:V
        n = norm(W[:,i]-W[:,j])
        l[j] = (n, j)
    end

    sort!(l, by = x -> x[1])
    return [ x[2] for x in l[2:r] ]
end

function similaire(W,s::String,r)
    similaire(W,trad(s),r)
end

function similaire(W,v::Vector{Float64},r)

    l = []
    d = Dict{AbstractFloat,Int8}()

    for j in 1:V
        n = norm(v-W[:,j])
        append!(l,[n])
        d[n] = j
    end

    sort!(l)
    L = Array{Int64,1}()
    for i in 1:r
        append!( L ,[ d[l[i]] ] )
    end
    return(L)
end

function trad(i::Int)
    return(vocabulaire[i])
end

function trad(mot::String)
    findfirst(isequal(mot),vocabulaire)
end

function trad(veci::Array{Int})
    return([trad(i) for i in veci])
end

function trad(vecs::Array{String})
    return([trad(s) for s in vecs])
end

function analogy(W,v1::String,v2::String,v3::String;n::Int=5)

    i = trad(v1)
    j = trad(v2)
    k = trad(v3)
    v4 = W[:,i]-W[:,j]+W[:,k]
    similaire(W,v4,n)

end

function analogy(W,i::Int,j::Int,k::Int;n::Int=5)

    v4 = W[:,i]-W[:,j]+W[:,k]
    similaire(W,v4,n)

end


simi1 = similaire(WA1,"CCT",10);
simi12 = similaire(WA12,"CCT",10);
simi2 = similaire(WA2,"CTT",10);
simi3 = similaire(WB,"CTT",10);
simi1bis = similaire(WA1bis,"CTT",10);

[simi1 simi12 simi1bis simi2 simi3]

trad(simi1) 
trad(simi12)
trad(simi2) 
trad(simi3)
trad(simi1bis)

trad(analogy(WA1,"CCT","TTG","AAA"))

function remplace(mot::String)
    n = length(mot)
    nouveau_mot = ""
    for i in 1:n

        lettre = mot[i]

        if lettre == 'A'
            nouveau_mot *= "U"
        end
        if lettre == 'T'
            nouveau_mot *= "A"
        end
        if lettre == 'G'
            nouveau_mot *= "C"
        end
        if lettre == 'C'
            nouveau_mot *= "G"
        end
    end
    nouveau_mot
end

remplace("ATGC")


vocab = [remplace(codon) for codon in vocabulaire]
#= vocab = [replace(codon,'A'=>'U','T'=>'A','G'=>'C','C'=>'G') for codon in vocabulaire]
replace("AAT","A"=>"U","T"=>"A","G"=>"C","C"=>"G")
replace("AAT",'A'=>'U','T'=>'A','G'=>'C','C'=>'G')

replace([1,2,3,4],1=>2,3=>0) =#


scatter(Y2[1,:],Y2[2,:], hover = vocab)
scatter(Y3[1,:],Y3[2,:], hover = vocab,
series_annotations = [text("   "*voc, Plots.font("Sans", 8), :left) 
 for voc in vocab])


