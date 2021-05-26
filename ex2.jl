using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using Plots
using Statistics

xmax = 1
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

        if i == 1
            r = texte[i+1:sup]

            for j in 1:length(r)
                dic[mot][r[j]]+=poids(j)
            end

        elseif i == n
            l = texte[inf:i-1]

            for j in 1:length(l)
                dic[mot][l[j]]+=poids(window - j + 1)
            end

        else
            l = texte[inf:i-1]
            r = texte[i+1:sup]

            for j in 1:length(l)
                dic[mot][l[j]]+=poids(window - j + 1)
            end
            for j in 1:length(r)
                dic[mot][r[j]]+=poids(j)
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
    shift = Float64(0.5)
    Glove(
        (rand(Float64, vecsize, vocabsize) .- shift) ./ Float64(vecsize + 1),
        (rand(Float64, vecsize, vocabsize) .- shift) ./ Float64(vecsize + 1),
        (rand(Float64, vocabsize) .- shift) ./ Float64(vecsize + 1),
        (rand(Float64, vocabsize) .- shift) ./ Float64(vecsize + 1) )
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
            if m == 0
                continue
            end
            
            wj = g.W_ctx[:,j]
            bj = g.b_ctx[j]

            S += f(m)*( dot(wi,wj) + bi + bj - log(m) )^2
        end
    end
    S
end

cost2(M2)

function update_loss!(mat)
    push!(train_loss, cost2(mat))
    println("train loss = $(train_loss[end])")
end

# comparer differents opts

#= g = Glove(64,10)
p = params(g)
opt = Flux.Optimise.ADAGrad()
Flux.train!(cost2, p, Iterators.repeated(M2, 200), opt; cb = update_loss!)

img = plot(1:length(train_loss), train_loss, xlabel="epochs", ylabel="loss", label="ADAGrad")

g = Glove(64,10)
train_loss = [cost2(M2)]
p = params(g)
opt = Flux.Optimise.Descent(0.01)
Flux.train!(cost2, p, Iterators.repeated(M2, 200), opt; cb = update_loss!)

plot!(1:length(train_loss), train_loss, xlabel="epochs", ylabel="loss", label="Descent")

g = Glove(64,10)
train_loss = [cost2(M2)]
p = params(g)
opt = Flux.Optimise.ADAM()
Flux.train!(cost2, p, Iterators.repeated(M2, 200), opt; cb = update_loss!)

plot!(1:length(train_loss), train_loss, xlabel="epochs", ylabel="loss", label="ADAM")

@time Flux.train!(cost2, p, Iterators.repeated(M2, 100), opt)

savefig(img,"opt.png") =#

# lire les sequences

function clean(sequence)
    replace(sequence, r"[\n\r\s]"=>"")
end

function lire(name)
    A1 = open("A01010101.txt") do file
        read(file, String)
    end
    clean(A1)
end

A1 = lire("A01010101.txt")
A2 = lire("A01010101.txt")
B = lire("B07020101.txt")


function vecSequence(sequence)
    l = []
    n = length(sequence)
    n -= n%3
    for i in 1:(Int(n/3)-1)
        append!(l,[sequence[i:i+2]])
    end
    l
end

dico_A1 = create_dic()
dico_A2 = create_dic()
dico_B = create_dic()

co_oc!(dico_A1,vecSequence(A1),5)
co_oc!(dico_A2,vecSequence(A2),5)
co_oc!(dico_B,vecSequence(B),5)

MA1 = matrice(dico_A1)
MA2 = matrice(dico_A2)
MB = matrice(dico_B)


g = Glove(64,10)
p = params(g)
opt = Flux.Optimise.ADAGrad()
Flux.train!(cost2, p, Iterators.repeated(MA1, 50), opt)

WA1_main = g.W_main
WA1_ctx = g.W_ctx

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

#= gr()
plot(Y[1,:],Y[2,:], seriestype = :scatter)
plot(Y[1,:],Y[2,:], seriestype = :scatter, series_annotations = Plots.series_annotations(vocabulaire, Plots.font("Sans", 7)))

img = plot(Y[1,:],Y[2,:], marker=(2,0.2,:black), seriestype = :scatter, series_annotations = [text("   "*voc, Plots.font("Sans", 2), :left) for voc in vocabulaire])
Plots.savefig(img,"A1.pdf")

scatter!(linspace(2,8,6),rand(6),marker=(50,0.2,:orange),series_annotations=["series","annotations","map","to","series",text("data",:green)]) =#

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
 WB = WB_main+WB_main;

 norm(WA1[:,1])
 norm(WA2[:,1])
 norm(WB[:,1])

m1 = mean([norm(WA1[:,i]-WA2[:,i]) for i in 1:V])
m2 = mean([norm(WA1[:,i]-WB[:,i]) for i in 1:V])

m1 = mean(x->x^2,[norm(WA1[:,i])-norm(WA2[:,i]) for i in 1:V])
m2 = mean(x->x^2,[norm(WA1[:,i])-norm(WB[:,i]) for i in 1:V])

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
s(WA1,WB)
s(WA2,WB)

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


simi1 = similaire(WA1,"CTG",7); #CCT
simi2 = similaire(WA2,"GCT",7);
simi3 = similaire(WB,"GCT",7); #CCT

trad(simi1)
trad(simi2)
trad(simi3)

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
 
