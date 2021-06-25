# Glove model
struct Model
    W_main::Matrix{Float64}
    W_ctx::Matrix{Float64}
    b_main::Vector{Float64}
    b_ctx::Vector{Float64}
    W_main_grad::Matrix{Float64}
    W_ctx_grad::Matrix{Float64}
    b_main_grad::Vector{Float64}
    b_ctx_grad::Vector{Float64}
end

"""
Each vocab word in associated with a word vector and a context vector.
The paper initializes the weights to values [-0.5, 0.5] / vecsize+1 and
the gradients to 1.0.
The +1 term is for the bias.
"""
function Model(vocabsize::Int, vecsize::Int)
    shift = Float64(0.5)
    Model(
        (rand(Float64, vecsize, vocabsize) .- shift) ,
        (rand(Float64, vecsize, vocabsize) .- shift) ,
        (rand(Float64, vocabsize) .- shift) ,
        (rand(Float64, vocabsize) .- shift) ,
        ones(Float64, vecsize, vocabsize),
        ones(Float64, vecsize, vocabsize),
        ones(Float64, vocabsize),
        ones(Float64, vocabsize),
    )
end

#= W_main = (rand(Float64, 64, 10) .- 0.5) ./ Float64(64 + 1)
size(W_main, 2) =#

function matrice2(n::Int)
    m = zeros(Float64,n,n)
    for i in 1:n
        for j in i:n
            r = rand(1:6)
            if r > 4
                v = max(0,200*randn()+1000)
                m[i,j] = v
                m[j,i] = v
            elseif r == 3
                v = max(0,200*randn()+500)
                m[i,j] = v
                m[j,i] = v
            end
        end
    end
    m
end

Voc = 4^6

Mrand = matrice2(Voc)

function adagrad!(
  m::Model;
  #= mat::Vector{Vector{Int64}}; =#
  epochs::Int=1,
  lrate=Float64(1e-1),
  xmax::Int=1000,
  alpha=Float64(0.75))

    mat = Mrand
    # store the cost per iteration
    costs = zeros(Float64, epochs)
    vecsize = size(m.W_main, 1)
    V = length(mat[:,1])

    for n = 1:epochs
        for i = 1:V
            for j = 1:V

                val = mat[i,j] # value

                if val == 0
                    continue
                end

                dif = m.b_main[i] + m.b_ctx[j] - log(val)

                for k = 1:vecsize
                dif += m.W_main[k,i] * m.W_ctx[k,j]
                end #produit scalaire

                fdif = ifelse(val < xmax, (val / xmax) ^ alpha, Float64(1)) * dif
                
                costs[n] = Float64(0.5) * fdif * dif


                # Adaptive learning gradient updates
                fdif *= lrate
                for k = 1:vecsize
                    tm = fdif * m.W_ctx[k,j]
                    #= println("tm = $(tm)") =#
                    tc = fdif * m.W_main[k,i]
                    #= println("tc = $(tc)") =#
                    m.W_main[k,i] -= tm / sqrt(m.W_main_grad[k,i])
                    #= println("m.W_main[k,i] = $(m.W_main[k,i])") =#
                    m.W_ctx[k,j] -= tc / sqrt(m.W_ctx_grad[k,j])
                    #= println("m.W_ctx[k,j] = $(m.W_ctx[k,j])") =#
                    m.W_main_grad[k,i] += tm * tm
                    m.W_ctx_grad[k,j] += tc * tc
                end

                # bias updates
                m.b_main[i] -= fdif / sqrt(m.b_main_grad[i])
                m.b_ctx[j] -= fdif / sqrt(m.b_ctx_grad[j])
                fdif *= fdif
                m.b_main_grad[i] += fdif
                m.b_ctx_grad[j] += fdif

            end
        end
    end
    costs
end

function dif(m1::Model,m2::Model)
    Model(m1.W_main-m2.W_main,
    m1.W_ctx-m2.W_ctx,
    m1.b_main-m2.b_main,
    m1.b_ctx-m2.b_ctx,
    m1.W_main_grad-m2.W_main_grad,
    m1.W_ctx_grad-m2.W_ctx_grad,
    m1.b_main_grad-m2.b_main_grad,
    m1.b_ctx_grad-m2.b_ctx_grad)
end

m = Model(Voc,10)
@time adagrad!(m)
