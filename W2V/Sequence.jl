
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

function clean(sequence)
    replace(sequence, r"[\n\r\s]"=>"")
end

function lire(name)
    f = open(name) do file
        read(file, String)
    end
    clean(f)
end

function vecSequence(sequence) # non-overlapping : 3 listes de 3-mers 
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

function vecSequence2(sequence) # non-overlapping: 1 liste de 3-mers
    l1 = []
    n = length(sequence)
    i = 1
    while i+2<n+1
        append!(l1,[sequence[i:i+2]])
        i+=3
    end
    return l1
end

function vecSequence3(sequence) # overlapping: 1 liste de 3-mers
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