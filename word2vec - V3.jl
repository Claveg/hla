using Base: String

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

function clean(sequence)
    replace(sequence, r"[\n\r\s]"=>"")
end

function lire(name)
    f = open(name) do file
        read(file, String)
    end
    clean(f)
end

function vecSequence(sequence)
    l1 = []
    n = length(sequence)
    i = 1
    while i+2<n+1
        append!(l1,[sequence[i:i+2]])
        i+=3
    end

    l2 = []
    i = 2
    while i+2<n+1
        append!(l2,[sequence[i:i+2]])
        i+=3
    end

    l3 = []
    i = 3
    while i+2<n+1
        append!(l3,[sequence[i:i+2]])
        i+=3
    end
    return (l1,l2,l3)
end

#= Stest = "AAABBBCCCDD"
(l1,l2,l3) = vecSequence(Stest) =#

function co_oc(l,window)
    n = length(l)
    X = []
    for i in 1:n
        inf = max(1,i-window)
        sup = min(n, i+window)

        vec = zeros(V)
        for j in inf:sup            
            if j!=i
                vec += onehot(l[j], vocabulaire)
            end
        end
        append!(X,[vec])
    end
    return batch(X)
end

function create_batch(name::String, window::Int)
    A = lire(name)
    (l1,l2,l3) = vecSequence(A)

    Y1 = batch( [onehot(codon,vocabulaire) for codon in l1] )
    Y2 = batch( [onehot(codon,vocabulaire) for codon in l2] )
    Y3 = batch( [onehot(codon,vocabulaire) for codon in l3] )

    Y = [Y1 Y2 Y3]

    x1 = co_oc(l1,window)
    x2 = co_oc(l2,window)
    x3 = co_oc(l3,window)

    X = [x1 x2 x3]

    return (X,Y)
end

data = create_batch("A01010101.txt",5)

model = Wv(V,10)
ps = params(model)
opt = ADAGrad()

cst(data...)
cst(data[1][:,1],data[2][:,1])

@time train!(cst, ps, [data], opt)

loss = []

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

W_A1 = model.W
W1_A1 = model.W1

model = Wv(V,10)
ps = params(model)
data = create_batch("A01010102.txt",5)

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

W_A2 = model.W
W1_A2 = model.W1

model = Wv(V,10)
ps = params(model)
data = create_batch("B07020101.txt",5)

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

W_B = model.W
W1_B = model.W1

model = Wv(V,10)
ps = params(model)
data = create_batch("A01010101.txt",5)

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

W_A1bis = model.W
W1_A1bis = model.W1

simiA1 = similaire(W_A1,"CCT",10);
simiA1bis = similaire(W_A1bis,"CCT",10);
simiA2 = similaire(W_A2,"CCT",10);
simiB = similaire(W_B,"CTT",10);

[simiA1 simiA1bis simiA2 simiB]

data = create_batch("A01010101.txt",5)

model = Wv(V,20)
ps = params(model)
opt = ADAGrad()

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

W_A1 = model.W
W1_A1 = model.W1

model = Wv(V,20)
ps = params(model)
data = create_batch("A01010102.txt",5)

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

W_A2 = model.W
W1_A2 = model.W1

model = Wv(V,20)
ps = params(model)
data = create_batch("B07020101.txt",5)

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

W_B = model.W
W1_B = model.W1

model = Wv(V,20)
ps = params(model)
data = create_batch("A01010101.txt",5)

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

W_A1bis = model.W
W1_A1bis = model.W1

simiA1 = similaire(W_A1+W1_A1',"CCT",10);
simiA1bis = similaire(W_A1bis+W1_A1bis',"CCT",10);
simiA2 = similaire(W_A2+W1_A2',"CCT",10);
simiB = similaire(W_B+W1_B',"CTT",10);

[simiA1 simiA1bis simiA2 simiB]

function cosined(a,b)
    return( dot(a,b)/(norm(a)*norm(b)) )
end

function similaire(W,i::Int64,r)
    l = [ (-1.0, 0) for i in 1:V ]
    
    for j in 1:V
        n = cosined(W[:,i],W[:,j])
        l[j] = (n, j)
    end

    sort!(l, by = x -> x[1], rev=true)
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

