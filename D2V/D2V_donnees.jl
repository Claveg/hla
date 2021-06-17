using Flux: onehot, batch

include("D:\\Mines\\Stage IRSL\\hla\\W2V\\Sequence.jl")
include("D:\\Mines\\Stage IRSL\\hla\\W2V\\Donnees.jl")


function create_hdy(seq, allele, window::Int)
    
    (l1,l2,l3) = vecSequence(seq)

    Y1 = batch( [onehot(codon,vocabulaire) for codon in l1] )
    Y2 = batch( [onehot(codon,vocabulaire) for codon in l2] )
    Y3 = batch( [onehot(codon,vocabulaire) for codon in l3] )

    Y = [Y1 Y2 Y3]

    x1 = co_oc(l1,window)
    x2 = co_oc(l2,window)
    x3 = co_oc(l3,window)

    d1 = [onehot(allele, nom_allele[1:Vdoc]) for i in 1:length(l1) ]
    d2 = [onehot(allele, nom_allele[1:Vdoc]) for i in 1:length(l2) ]
    d3 = [onehot(allele, nom_allele[1:Vdoc]) for i in 1:length(l3) ]

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

        X = hcat(X,x)
        D = hcat(D,d)
        Y = hcat(Y,y)

    end

    (X,D,Y)

end