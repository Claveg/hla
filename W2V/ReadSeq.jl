
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

V = length(vocabulaire)

sequence = "AAATTTGGGCCCAAA"^100

function 
function readSeq(sequence::String)

    n = length(sequence)
    M1 = spzeros(V,n÷3)
    i = 1    

    while i+2<n+1
        codon = sequence[i:i+2]
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

using Flux

B = Flux.onehotbatch([:b, :a, :b], [:a, :b, :c])

M1 = rand(64,1000);
M2 = rand(64,1000);

M = Iterators.repeated(M1,140)

@elapsed [M1 M2]
@elapsed hcat(M1,M2)

@elapsed Flux.batch(M)

oh = Flux.onehot(:1, [:1,:2,:3])

Vd = collect(1:14000)

S = 0
for i in 1:1000

    S+= @elapsed d1 = Flux.batch([Flux.onehot(1200, Vd) for i in 1:1000 ]) # 2.9 e-3

end

S2 = 0
for i in 1:1000
    S2+= @elapsed d1 = Flux.onehotbatch(Iterators.repeated(1200,1000), Vd) # 3.6 e-3
end


O = Iterators.repeat(oh,1,10)
Oh = Iterators.repeated(oh,10)
Oooh = Flux.batch(Oh)


@elapsed 1+1 #1e-7 probablement moins ?

b = Flux.onehot(1, collect(1:64))

M = rand(100,100)

S = 0
for i in 1:1000
    S+= @elapsed b+b # 7e-7
end
S/1000

S = 0
t = 0
for i in 1:1000
    i = rand(1:100)
    j = rand(1:100)
    S+= @elapsed a = M[i,j] # 1e-7
    t+=a
end
S/1000