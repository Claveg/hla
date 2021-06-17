# creer une liste de toutes les allèles de IMGT

struct Allele
    id
    nom
    sequence
end

function create_db(fname)
    
    db = Allele[]

    ide = nothing
    name = nothing
    seq = nothing

    for line in eachline(fname)

        if line[1] == '>'
            if (name != nothing)
                a = Allele(ide,name,seq)
                push!(db,a)
            end

            m = match( r"HLA\d+" , line)
            ide = m.match

            m = match( r"\S+\*\d+[:\d+]*", line )
            name = m.match

            seq = ""

        else
            seq *= line
        end

    end
    db
end

db1 = create_db(raw"Sequences\hla_gen.fasta");

n = length(db1)

nom_allele = Array{String}(undef, n)

for i in 1:n
    a = db1[i]
    nom_allele[i] = a.nom
end