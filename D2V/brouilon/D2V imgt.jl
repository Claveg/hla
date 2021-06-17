using DelimitedFiles

V = 64
Vdoc = 10
hiddoc = 10
hidword = 10
c = 5

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

#= (x,d,y) = create_hdy(db1[1].sequence,db1[1].nom,10) =#

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

db2 = copy(db1[1:Vdoc])

@time (X,D,Y) = imgt(db2,c)

#= gpu(X)
gpu(Y)
gpu(Z) =#

data = (X,D,Y)

word2 = Dense(V,hidword; bias = false)
doc2 = Dense(Vdoc,hiddoc; bias = false)
h2(data) =  [word2(data[1]); doc2(data[2])]
y2 = Dense(hidword + hiddoc, V; bias = false)

D2V = Chain( h2 , y2 , softmax) #= |> gpu =#
ps = Flux.params(D2V)

cost(x_w,x_d,y) = crossentropy( D2V([x_w,x_d]), y)

lost = []

function update_loss!(ls)
    push!(ls, cost(data...))
end

opt = ADAGrad()

@time train!(cost, ps, Iterators.repeated(data,20), opt; cb = () -> update_loss!(lost))

plot(lost)

(lost[1]-lost[end])/lost[1]

model2 = Dv(V,Vdoc,hidword,hiddoc)
ps2 = Flux.params(model2)
opt = ADAGrad()

lost = []
@time train!(cst2, ps2, Iterators.repeated(data,19), opt; cb = () -> update_loss2!(lost))

