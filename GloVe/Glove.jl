using Flux

struct Glove
    W_main::Matrix{Float64}
    W_ctx::Matrix{Float64}
    b_main::Vector{Float64}
    b_ctx::Vector{Float64}
end

function oness(p, q)
    m = zeros(Float64, p, q)
    for j in 1:q
        k = 1
        for i in 1:p
            m[i,j] = k
            k += 1
        end
    end
    return m
end

function Glove(vocabsize::Int, vecsize::Int)

    Glove(
        rand(Float64, vecsize, vocabsize).-0.5,
        rand(Float64, vecsize, vocabsize).-0.5,
        rand(Float64, vocabsize).-0.5,
        rand(Float64, vocabsize).-0.5)

end

Flux.@functor Glove

f(x) = min( (x/xmax)^0.75 , 1)

function cost(mat::Matrix{Float64})

    S = 0
    for i in 1:V

        wi = g.W_main[:,i]
        bi = g.b_main[i]

        for j in 1:V
            m = mat[i,j]
            if m != 0
                wj = g.W_ctx[:,j]
                bj = g.b_ctx[j]

                S += f(m)*( dot(wi,wj) + bi + bj - log(m) )^2
            end
        end
    end
    S
end

function update_loss!(mat)
    push!(train_loss, cost(mat))
    println("train loss = $(train_loss[end])")
end