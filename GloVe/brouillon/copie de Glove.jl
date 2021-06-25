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
function Model(vocabsize::Int, vecsize::Int; rd::Bool=true)
    shift = Float64(0.5)
    if rd
        Model(
            (rand(Float64, vecsize, vocabsize) .- shift) ./ Float64(vecsize + 1),
            (rand(Float64, vecsize, vocabsize) .- shift) ./ Float64(vecsize + 1),
            (rand(Float64, vocabsize) .- shift) ./ Float64(vecsize + 1),
            (rand(Float64, vocabsize) .- shift) ./ Float64(vecsize + 1),
            ones(Float64, vecsize, vocabsize),
            ones(Float64, vecsize, vocabsize),
            ones(Float64, vocabsize),
            ones(Float64, vocabsize)
        )
    else
        Model(
            init1,
            init2,
            init3,
            init4,
            ones(Float64, vecsize, vocabsize),
            ones(Float64, vecsize, vocabsize),
            ones(Float64, vocabsize),
            ones(Float64, vocabsize)
        )
    end
end


#= W_main = (rand(Float64, 64, 10) .- 0.5) ./ Float64(64 + 1)
size(W_main, 2) =#

function adagrad!(
  m::Model;
  #= mat::Vector{Vector{Int64}}; =#
  epochs::Int=50,
  lrate=Float64(1e-2),
  xmax::Int=20,
  alpha=Float64(0.75))

    mat = MA1
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
                    tc = fdif * m.W_main[k,i]
                    m.W_main[k,i] -= tm / sqrt(m.W_main_grad[k,i])
                    m.W_ctx[k,j] -= tc / sqrt(m.W_ctx_grad[k,j])
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

m = Model(64,10; rd = false)
adagrad!(m)

gm1 = m.W_main
gc1 = m.W_ctx
gbm1 = m.b_main
gbc1 = m.b_ctx

gdm1 = m.W_main-oups
gdc1 = m.W_ctx-oups2
gdbm1 = m.b_main-oups3
dbc1 = m.b_ctx-oups4


m.W_main;
plot(adagrad!(m))
m.W_main;
@time adagrad!(m)
m.W_main_grad;

#= using MultivariateStats

pc = MultivariateStats.fit(PCA, m.W_main; maxoutdim=2)
Y = transform(pc, m.W_main)

gr() 
plot(Y[1,:],Y[2,:], seriestype = :scatter) =#

