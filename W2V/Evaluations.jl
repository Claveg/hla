using LinearAlgebra: include
using DelimitedFiles
using LinearAlgebra

include("Sequence.jl")
include("donnees.jl")

# le modele s'appelle ici "model"

function cosined(a,b)
    return( dot(a,b)/(norm(a)*norm(b)) )
end

function similaire(W,i::Int64,r) # r plus proches vecteurs de i dans W (cosined)
    l = [ (-1.0, 0) for i in 1:V ]
    
    for j in 1:V
        n = cosined(W[:,i],W[:,j])
        l[j] = (n, j)
    end

    sort!(l, by = x -> x[1], rev=true)
    return [ x[2] for x in l[2:r+1] ]
end

function similaire2(W,i::Int64,r) # r plus proches vecteurs de i dans W (norm)
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


function eval(data) # norm( y - pred(x) ) 
    n,p = size(data[1])
    S = 0
    for i in 1:p
        y = data[2][:,i]
        pred = model(data[1][:,i])
        S += norm(y - pred)
    end
    return S/p
end

function eval2(data) # pred(x) == y ?
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

function prediction(y,s1,s2) # renvoie les indices les plus probables pour pred(x)
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

function eval3(data,s1,s2) # y ∈ prediction(x) ?
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

function evalsimi(W1,W2,n) # similaire(W1,codon,n) .== similaire(W2,codon,n)

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

# calcule eval1,2,3 et evalsimi pour différentes tailles de contexte (range)

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

function sauvegarde(name,texte,e1,e2,e3,ms) #sauvegarde les resultats de tcontext
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