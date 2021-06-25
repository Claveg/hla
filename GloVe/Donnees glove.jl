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