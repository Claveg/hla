using Flux
using Flux: onehot, batch, onecold
using SparseArrays

include("Sequence.jl")

#= function co_oc(l,window)
    n = length(l)
    X = spzeros(V,n)

    for i in 1:n
        inf = max(1,i-window)
        sup = min(n, i+window)

        for j in inf:sup            
            if j!=i
                X[:,i] += onehot(l[j], vocabulaire)
            end
        end
    end
    return X
end =#

function co_oc(l,window)
    n = length(l)
    X = spzeros(V,n-50)

    for i in 1:n-50
        if l[i] != 3 * "N"

        inf = max(1,i-window)
        sup = min(n, i+window)

        for j in inf:sup 
            if l[j] != 3 * "N"
                if j!=i
                    ind = trad(l[j])
                    X[ind,i] += 1
                end
        end
    end
    return X
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