using Random
using LinearAlgebra
using Flux
using Flux.Optimise
using DelimitedFiles
using Plots

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
    a = [vocabulaire[1:V] for i in 1:n]
end

function matrice(dic)
    m = zeros(Float64,V,V)
    for i in 1:V
        for j in i:V
            v = dic[vocabulaire[i]][vocabulaire[j]]
            m[i,j] = v
            m[j,i] = v
        end
    end
    m
end

struct Glove
    W_main::Matrix{Float64}
    W_ctx::Matrix{Float64}
    b_main::Vector{Float64}
    b_ctx::Vector{Float64}
end

function oness(p, q)
    m = zeros(Float64, p, q)
    for j in 1:q
        k = 1
        for i in 1:p
            m[i,j] = k
            k += 1
        end
    end
    return m
end

init1 = (rand(10, 64).-0.5).*10
oups = init1[:,:]
init2 = (rand(10, 64).-0.5).*10
oups2 = init2[:,:]
init3 = (rand(64).-0.5).*10
oups3 = init3[:,:]
init4 = (rand(64).-0.5).*10
oups4 = init4[:,:]

function Glove(vocabsize::Int, vecsize::Int; rd::Bool=true)
    if rd
        Glove(
            rand(Float64, vecsize, vocabsize).-0.5,
            rand(Float64, vecsize, vocabsize).-0.5,
            rand(Float64, vocabsize).-0.5,
            rand(Float64, vocabsize).-0.5)
    else
        Glove(
            init1,
            init2,
            init3,
            init4)
    end
end

Flux.@functor Glove

f(x) = min( (x/xmax)^0.75 , 1)

g = Glove(64,10)

m = g.W_main[:,:]
c = g.W_ctx[:,:]
bm = g.b_main[:]
bc = g.b_ctx[:]

p = params(g)
opt = Flux.Optimise.ADAGrad(0.01)
Flux.train!(cost, p, Iterators.repeated(Mf1, 50), opt)

m1 = g.W_main
c1 = g.W_ctx
bm1 = g.b_main
bc1 = g.b_ctx

dm1 = g.W_main-oups
dc1 = g.W_ctx-oups2
dbm1 = g.b_main-oups3
dbc1 = g.b_ctx-oups4


function sauvegarde(name,m,c,bm,bc,dm,dc,dbm,dbc)
    open("$(name).txt", "w") do io
        write(io, "\n\nW_main\n\n")
        writedlm(io, m)

        write(io, "\n\nW_ctx\n\n")
        writedlm(io, c)

        write(io, "\n\nb_main\n\n")
        writedlm(io, bm)
        
        write(io, "\n\nb_ctx\n\n")
        writedlm(io, bc)
        
        write(io, "\n\nDW_main\n\n")
        writedlm(io, dm)

        write(io, "\n\nDW_ctx\n\n")
        writedlm(io, dc)

        write(io, "\n\nDb_main\n\n")
        writedlm(io, dbm)
        
        write(io, "\n\nDb_ctx\n\n")
        writedlm(io, dbc)

    end
end

sauvegarde("mp",m1,c1,bm1,bc1,dm1,dc1,dbm1,dbc1)

4^6


function cost(mat::Matrix{Float64})

    S = 0
    for i in 1:V

        wi = g.W_main[:,i]
        bi = g.b_main[i]

        for j in 1:V
            m = mat[i,j]
            if m != 0
                wj = g.W_ctx[:,j]
                bj = g.b_ctx[j]

                S += f(m)*( dot(wi,wj) + bi + bj - log(m) )^2
            end
        end
    end
    S
end

function update_loss!(mat)
    push!(train_loss, cost(mat))
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
Atest = lire("seqtest.txt")
A2 = lire("A01010102.txt")
B = lire("B07020101.txt")

function vecSequence(sequence)
    l = []
    n = length(sequence)
    i = 1
    while i+2<n+1
        append!(l,[sequence[i:i+2]])
        i+=3
    end
    l
end

dico_test = create_dic()
co_oc!(dico_test,vecSequence(Atest),3; poids = x->4-x)
Mtest = matrice(dico_test)

dico_A1 = create_dic()
dico_A2 = create_dic()
dico_B = create_dic()

co_oc!(dico_A1,vecSequence(A1),5; poids = x->6-x)
co_oc!(dico_A2,vecSequence(A2),5; poids = x->6-x)
co_oc!(dico_B,vecSequence(B),5; poids = x->6-x)

MA1 = matrice(dico_A1)
MA2 = matrice(dico_A2)
MB = matrice(dico_B)

function proba(m)
    mat = m[:,:]
    for i in 1:V
        mat[i,:] = mat[i,:]/sum(mat[i,:])
    end
    mat
end

function test_meth(P)
    Imin = zeros(V)
    Imax = zeros(V)
    Mat = zeros(V)
    for j in 1:V
        M = minimum(P) - 1
        m = maximum(P) + 1
        imax = 0
        imin = 0 
        for i in 1:V
            val = P[i, j]
            if val > M
                M = val
                imax = i
            end 
            if val < m && val > 1e-5
                m = val
                imin = i
            end
        end
        Mat[j] = M / m
        Imin[j] = imin
        Imax[j] = imax
    end
    (Mat, Imin, Imax)
end

norm(MA1-MA2)
norm(MA1-MB)

g = Glove(64,10)
p = params(g)
opt = Flux.Optimise.ADAGrad()
Flux.train!(cost, p, Iterators.repeated(MA1, 50), opt)

WA1_main = g.W_main
WA1_ctx = g.W_ctx

g = Glove(64,10)
p = params(g)
opt = Flux.Optimise.ADAGrad()
Flux.train!(cost, p, Iterators.repeated(MA1, 50), opt)

WA1_main2 = g.W_main
WA1_ctx2 = g.W_ctx

g = Glove(64,10)
p = params(g)
Flux.train!(cost, p, Iterators.repeated(MA2, 50), opt)

WA2_main = g.W_main
WA2_ctx = g.W_ctx

g = Glove(64,10)
p = params(g)
Flux.train!(cost, p, Iterators.repeated(MB, 50), opt)

WB_main = g.W_main
WB_ctx = g.W_ctx

# essais pour comparer les embeddings des sequences 

 WA1 = WA1_main+WA1_ctx;
 WA2 = WA2_ctx+WA2_main;
 WA12 = WA1_main2+WA1_ctx2;
 WB = WB_main+WB_main;

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
s(WA1,WA1)
s(WA1,WB)
s(WA2,WB)
s(WA1,WA12)

function cosined(a,b)
    return( dot(a,b)/(norm(a)*norm(b)) )
end

function similaire(W,i::Int64,r)
    l = [ (-1.0, 0) for i in 1:V ]
    
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

[simi1 simi12 simi2 simi3]

# test avec initialisation fixe
 
g = Glove(64,10; rd = false)
p = params(g)
opt = Flux.Optimise.ADAGrad()
Flux.train!(cost, p, Iterators.repeated(MA1, 50), opt)

WA1_main = g.W_main
WA1_ctx = g.W_ctx

g = Glove(64,10; rd = false)
p = params(g)
opt = Flux.Optimise.ADAGrad()
Flux.train!(cost, p, Iterators.repeated(MA1, 50), opt)

WA1_main2 = copy(g.W_main)
WA1_ctx2 = copy(g.W_ctx)

g = Glove(64,10; rd = false)
p = params(g)
Flux.train!(cost, p, Iterators.repeated(MA2, 50), opt)

WA2_main = copy(g.W_main)
WA2_ctx = copy(g.W_ctx)

g = Glove(64,10; rd = false)
p = params(g)
Flux.train!(cost, p, Iterators.repeated(MB, 50), opt)

WB_main = copy(g.W_main)
WB_ctx = copy(g.W_ctx)

WA1 = WA1_main+WA1_ctx;
WA2 = WA2_ctx+WA2_main;
WA12 = WA1_main2+WA1_ctx2;
WB = WB_main+WB_main;

simi1 = similaire(WA1,"CCT",10);
simi12 = similaire(WA12,"CCT",10);
simi2 = similaire(WA2,"CTT",10);
simi3 = similaire(WB,"CTT",10);

[simi1 simi12 simi2 simi3]

MB
cost(MB)
i=2
j=3
MB[i,j]
dot(g.W_main[:,i],g.W_ctx[:,j]) + g.b_main[i] + g.b_ctx[j]
log(MB[i,j])

cost(MB)
i=3
j=2
MB[i,j]
dot(g.W_main[:,i],g.W_ctx[:,j]) + g.b_main[i] + g.b_ctx[j]
log(MB[i,j])

 
# matrice plus "vide"

function faussematrice(n::Int)
    m = zeros(Float64,n,n)
    for i in 1:n
        for j in i:n
            r = rand(1:30)
            if r > 24
                v = max(0,200*randn()+200)*1000
                m[i,j] = v
                m[j,i] = v
            elseif r == 24
                v = max(0,200*randn()+1000)*1000
                m[i,j] = v
                m[j,i] = v
            end
        end
    end
    m
end

function faussematrice2(n::Int)
    m = zeros(Float64,n,n)
    for i in 1:n
        for j in i:n
            v = max(0,200*randn()+1)
            m[i,j] = v
            m[j,i] = v
        end
    end
    m
end

Mp1 = faussematrice2(V)

xmax = 1200
V = 4^3

Mf1 = faussematrice(V)
Mf2 = faussematrice(V)

g = Glove(V,10; rd = true)
p = params(g)
Flux.train!(cost, p, Iterators.repeated(Mf1,50), opt)

W1 = copy(g.W_main)+copy(g.W_ctx)
W1m = copy(g.W_main)
b1 = copy(g.b_main)+copy(g.b_ctx)

g = Glove(V,10; rd = true)
p = params(g)
Flux.train!(cost, p, Iterators.repeated(Mf2,50), opt)

W2 = copy(g.W_main)+copy(g.W_ctx)
W2m = copy(g.W_main)
b2 = copy(g.b_main)+copy(g.b_ctx)

g = Glove(V,10; rd = true)
p = params(g)
Flux.train!(cost, p, Iterators.repeated(Mf1,50), opt)

W3 = copy(g.W_main)+copy(g.W_ctx)
W3m = copy(g.W_main)
b3 = copy(g.b_main)+copy(g.b_ctx)

simi1 = similaire(W1,"CCT",10);
simi2 = similaire(W2,"CCT",10);
simi3 = similaire(W3,"CTT",10);

[simi1 simi3 simi2]

Mf1
cost(Mf1)
i=2
j=3
Mf1[i,j]
dot(g.W_main[:,i],g.W_ctx[:,j]) + g.b_main[i] + g.b_ctx[j]
log(Mf1[i,j])

ib=2
jb=2
Mf1[ib,jb]
dot(g.W_main[:,ib],g.W_ctx[:,jb]) + g.b_main[ib] + g.b_ctx[jb]
log(Mf1[ib,jb])


#augmenter V

xmax = 1200
V = 4^6

Mf = faussematrice(V)

g = Glove(V,5; rd = true)
p = params(g)

Wo = copy(g.W_main)+copy(g.W_ctx)
bo = copy(g.b_main)+copy(g.b_ctx)

#= @time Flux.train!(cost, p, [Mf], opt) =#

