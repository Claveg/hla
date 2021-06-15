using Base: String
using DelimitedFiles

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

function vecSequence2(sequence)
    l1 = []
    n = length(sequence)
    i = 1
    while i+2<n+1
        append!(l1,[sequence[i:i+2]])
        i+=3
    end
    return l1
end

function vecSequence3(sequence)
    l1 = []
    n = length(sequence)
    i = 1
    while i+2<n+1
        append!(l1,[sequence[i:i+2]])
        i+=1
    end
    return l1
end

function allele(n)
    a = bases[rand(1:4)]
    for i in 1:(n-1)
        a *= bases[rand(1:4)]
    end
    a
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

function co_oc2(l,window) # avec des "poids"
    n = length(l)
    X = []
    for i in 1:n
        inf = max(1,i-window)
        sup = min(n, i+window)

        vec = zeros(V)
        for j in inf:sup            
            if j!=i
                vec += 1/sqrt(abs(j-i)) * onehot(l[j], vocabulaire)
            end
        end
        append!(X,[vec])
    end
    return batch(X)
end

#= Stest = allele(42)
(l1,l2,l3) = vecSequence(Stest)

window = 4
x1 = co_oc2(l1,window)
x2 = co_oc2(l2,window)
x3 = co_oc2(l3,window)

x1[:,1][9] =#

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

function create_batch2(name::String, window::Int)
    A = lire(name)
    (l1,l2,l3) = vecSequence(A)

    Y1 = batch( [onehot(codon,vocabulaire) for codon in l1] )
    Y2 = batch( [onehot(codon,vocabulaire) for codon in l2] )
    Y3 = batch( [onehot(codon,vocabulaire) for codon in l3] )

    Y = [Y1 Y2 Y3]

    x1 = co_oc2(l1,window)
    x2 = co_oc2(l2,window)
    x3 = co_oc2(l3,window)

    X = [x1 x2 x3]

    return (X,Y)
end

function create_batch3(name::String, window::Int)
    A = lire(name)
    l1 = vecSequence2(A)

    Y1 = batch( [onehot(codon,vocabulaire) for codon in l1] )

    x1 = co_oc2(l1,window)

    return (x1,Y1)
end

function create_batch4(name::String, window::Int)
    A = lire(name)
    l1 = vecSequence3(A)

    Y = batch( [onehot(codon,vocabulaire) for codon in l1] )

    X = co_oc(l1,window)

    return (X,Y)
end

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
    return [ x[2] for x in l[2:r+1] ]
end

function similaire2(W,i::Int64,r)
    l = [ (-1.0, 0) for i in 1:V ]
    
    for j in 1:V
        n = norm(W[:,i]-W[:,j])
        l[j] = (n, j)
    end

    sort!(l, by = x -> x[1], rev=true)
    return [ x[2] for x in l[2:r] ]
end

function similaire2(W,s::String,r)
    similaire2(W,trad(s),r)
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

#= data = create_batch("A01010101.txt",5)

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

simiA1 = similaire(W_A1,"AGT",6);
simiA1bis = similaire(W_A1bis,"AGT",6);
simiA2 = similaire(W_A2,"AGT",6);
simiB = similaire(W_B,"AGT",6);

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

simiA1 = similaire(W_A1+W1_A1',"AGC",10);
simiA1bis = similaire(W_A1bis+W1_A1bis',"AGC",10);
simiA2 = similaire(W_A2+W1_A2',"AGC",10);
simiB = similaire(W_B+W1_B',"AGC",10);

[simiA1 simiA1bis simiA2 simiB] =#

function eval(data)
    n,p = size(data[1])
    S = 0
    for i in 1:p
        y = data[2][:,i]
        pred = model(data[1][:,i])
        S += norm(y - pred)
    end
    return S/p
end

function eval2(data)
    n,p = size(data[1])
    S = 0
    for i in 1:p
        y = data[2][:,i]
        yi = onecold(y)
        pred = model(data[1][:,i])
        predi = argmax(pred)
        S += (yi == predi)
    end
    return S/p
end

function prediction(y,s1,s2)
    ly = [ (y[i], i) for i in 1:length(y) ]

    sort!(ly, by = x -> x[1]; rev = true)

    k = 1
    l = [ly[1][2]]

    while ly[k][1] - ly[k+1][1] < s1 && ly[k+1][1] > s2
        k+=1
        push!(l,ly[k][2])
    end
    return l
end

function eval3(data,s1,s2)
    n,p = size(data[1])
    S = 0
    for i in 1:p
        y = data[2][:,i]
        yi = onecold(y)
        pred = model(data[1][:,i])
        predi = prediction(pred,s1,s2)
        S += (yi in predi)
    end
    return S/p
end


#= data = create_batch2("A01010101.txt",30)

model = Wv(V,20)
ps = params(model)
opt = ADAGrad()

loss = []

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

plot(loss)

W_A1 = model.W
W1_A1 = model.W1

println(eval(data))
println(eval2(data))
println(eval3(data,0.25,0.10))

model = Wv(V,20)
ps = params(model)
opt = ADAGrad()

loss = []

@time train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss))

W_A1bis = model.W
W1_A1bis = model.W1

println(eval(data))
println(eval2(data))
println(eval3(data,0.25,0.10))

simiA1 = similaire(W_A1+W1_A1',"AGA",5);
simiA1bis = similaire(W_A1bis+W1_A1bis',"AGA",5);

[simiA1 simiA1bis]

W1 = W_A1+W1_A1'
W2 = W_A1bis+W1_A1bis' =#


function evalsimi(W1,W2,n)

    S = zeros(V)
    k = 1

    for codon in vocabulaire
        simi1 = similaire(W1,codon,n)
        simi2 = similaire(W2,codon,n)
        s = 0
        for i in 1:n
            s+= (simi1[i] in simi2)
        end
        S[k] = s/n
        k+=1
    end
    S
end

e = evalsimi(W1,W2,5)
#= bar(e) =#
mean(e)


function eval(Ypred,Y)
    n,p = size(Ypred)
    S = 0
    for i in 1:p
        y = Y[:,i]
        pred = Ypred[:,i]
        S += norm(y - pred)
    end
    return S/p
end

function eval2(Ypred,Y)
    n,p = size(Ypred)
    S = 0
    for i in 1:p
        y = Y[:,i]
        yi = onecold(y)
        pred = Ypred[:,i]
        predi = argmax(pred)
        S += (yi == predi)
    end
    return S/p
end

function eval3(Ypred,Y,s1,s2)
    n,p = size(Ypred)
    S = 0
    for i in 1:p
        y = Y[:,i]
        yi = onecold(y)
        pred = Ypred[:,i]
        predi = prediction(pred,s1,s2)
        S += (yi in predi)
    end
    return S/p
end

function create_data(name,ctx,type)
    if type == 1
        data = create_batch(name,ctx)
        return data
    elseif type == 2
        data = create_batch2(name,ctx)
        return data
    elseif type == 3
        data = create_batch3(name,ctx)
        return data
    else
        data = create_batch4(name,ctx)
        return data
    end
end

function tcontext(range; hidsize = 20, train = 250, s1 = 0.25, s2 = 0.1, sim = 10, type=1 )

    n = length(range)

    eva1 = zeros(n,2)
    eva2 = zeros(n,2)
    eva3 = zeros(n,2)
    meansimi = zeros(n,V)

    j = 1

    for i in range

        data = create_data("A01010101.txt",i,type)

        model = Wv(V,hidsize)
        ps = params(model)
        opt = ADAGrad()

        cst = (x,y) -> crossentropy( softmax( model.W1 * (model.W * x) ), y)

        train!(cst, ps, Iterators.repeated(data,train), opt)

        W_A1 = model.W[:,:]
        W1_A1 = model.W1[:,:]

        Y = data[2]
        Ypred = model(data[1])

        x1 = eval(Ypred,Y)
        x2 = eval2(Ypred,Y)
        x3 = eval3(Ypred,Y,s1,s2)

        eva1[j,1] = x1
        eva2[j,1] = x2
        eva3[j,1] = x3
        
        model = Wv(V,hidsize)
        ps = params(model)
        opt = ADAGrad()

        cst = (x,y) -> crossentropy( softmax( model.W1 * (model.W * x) ), y)

        train!(cst, ps, Iterators.repeated(data,train), opt)

        W_A1bis = model.W[:,:]
        W1_A1bis = model.W1[:,:]

        Ypred = model(data[1])

        x4 = eval(Ypred,Y)
        x5 = eval2(Ypred,Y)
        x6 = eval3(Ypred,Y,s1,s2)

        eva1[j,2] = x4
        eva2[j,2] = x5
        eva3[j,2] = x6

        W_1 = copy(W_A1+W1_A1')
        W_2 = copy(W_A1bis+W1_A1bis')

        e = evalsimi(W_1,W_2,sim)
        meansimi[j,:] = e

        j+=1
    end

    return (eva1 , eva2 , eva3, meansimi)

end

#= @time (eva_1 , eva_2 , eva_3, mean_simi) = tcontext(1:30; train = 200, type = 3 , sim = 6)

plot(eva_1, label = "eva_1")
plot(eva_2, label = "eva_2")
plot(eva_3, label = "eva_3")

m_simi = [ mean(mean_simi[i,:]) for i in 1:length(mean_simi[:,1])]

plot(m_simi, label = "m_simi") =#

function sauvegarde(name,texte,e1,e2,e3,ms)
    open("$(name).txt", "w") do io

        write(io, "\n$(texte)\n\n")

        write(io, "\n\neva_1\n\n")
        writedlm(io, e1)

        write(io, "\n\neva_2\n\n")
        writedlm(io, e2)

        write(io, "\n\neva_3\n\n")
        writedlm(io, e3)

        write(io, "\n\nmean_simi\n\n")
        writedlm(io, ms)
    end
end

#= sauvegarde("type3_30","tcontext(1:30; train = 200, type = 3 , sim = 6)",eva_1,eva_2,eva_3,mean_simi) =#