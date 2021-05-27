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

Voc = 4^3

function adagrad!(
  m::Model;
  #= mat::Vector{Vector{Int64}}; =#
  epochs::Int=100,
  lrate=Float64(1e-1),
  xmax::Int=20,
  alpha=Float64(0.75))

    mat = rand(Float64,Voc,Voc)
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

wm1 = copy(m.W_main);
wb1 = copy(m.b_main);

plot(adagrad!(m))

wm2 = m.W_main
wb2 = m.b_main;

wm1-wm2
wb1-wb2


m.W_ctx
m.W_ctx_grad
@time adagrad!(m)
m.W_main_grad;

#= using MultivariateStats

pc = MultivariateStats.fit(PCA, m.W_main; maxoutdim=2)
Y = transform(pc, m.W_main)

gr() 
plot(Y[1,:],Y[2,:], seriestype = :scatter) =#

