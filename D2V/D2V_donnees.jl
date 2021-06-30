using Flux: onehot, batch

include("D:\\Mines\\Stage IRSL\\hla\\W2V\\Sequence.jl")
include("D:\\Mines\\Stage IRSL\\hla\\W2V\\Donnees.jl")


function create_hdy(seq, allele, window::Int)
    
    (l1,l2,l3) = vecSequence(seq)

    l = vcat(l1,l2,l3)
    Y = Flux.onehotbatch( l , vocabulaire)

    x1 = co_oc(l1,window)
    x2 = co_oc(l2,window)
    x3 = co_oc(l3,window)

    d1 = [onehot(allele, nom_allele) for i in 1:length(l1) ]
    d2 = [onehot(allele, nom_allele) for i in 1:length(l2) ]
    d3 = [onehot(allele, nom_allele) for i in 1:length(l3) ]

    X = [x1 x2 x3]

    D = [batch(d1) batch(d2) batch(d3)]

    return (X,D,Y)
end

function imgt(db,window)

    (X,D,Y) = create_hdy(db[1].sequence,db[1].nom,window)

    for i in 2:length(db)

        a = db[i]
        seq = a.sequence
        name = a.nom

        (x,d,y) = create_hdy(seq,name,window)

        X = [X x]
        Y = [Y y]
        D = [D d]        

    end

    (X,D,Y)

end

#= (X,D,Y) = create_hdy(db1[1].sequence,db1[1].nom,c)

seq = db1[1].sequence
window = c =#