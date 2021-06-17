using Flux
using Plots

#= include("D:\\Mines\\Stage IRSL\\hla\\W2V\\Sequence.jl") =#
include("D:\\Mines\\Stage IRSL\\hla\\W2V\\Donnees.jl")
include("D:\\Mines\\Stage IRSL\\hla\\W2V\\Word2vec.jl")
include("D:\\Mines\\Stage IRSL\\hla\\W2V\\Evaluations.jl")

data = create_batch("D:\\Mines\\Stage IRSL\\hla\\Sequences\\A01010101.txt",5)

function cst(x,y) 
    return Flux.crossentropy( model(x) , y)
end

model = Wv(V,10)

ps = params(model)
opt = ADAGrad()

cst(data...)
cst(data[1][:,1],data[2][:,1])

@time Flux.train!(cst, ps, [data], opt)

loss = []

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

W_A1 = model.W
W1_A1 = model.W1

model = Wv(V,10)
ps = params(model)
data = create_batch("D:\\Mines\\Stage IRSL\\hla\\Sequences\\A01010102.txt",5)

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

W_A2 = model.W
W1_A2 = model.W1

model = Wv(V,10)
ps = params(model)
data = create_batch("D:\\Mines\\Stage IRSL\\hla\\Sequences\\B07020101.txt",5)

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

W_B = model.W
W1_B = model.W1

model = Wv(V,10)
ps = params(model)
data = create_batch("D:\\Mines\\Stage IRSL\\hla\\Sequences\\A01010101.txt",5)

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

W_A1bis = model.W
W1_A1bis = model.W1

simiA1 = similaire(W_A1,"AGT",6);
simiA1bis = similaire(W_A1bis,"AGT",6);
simiA2 = similaire(W_A2,"AGT",6);
simiB = similaire(W_B,"AGT",6);

[simiA1 simiA1bis simiA2 simiB]

data = create_batch("D:\\Mines\\Stage IRSL\\hla\\Sequences\\A01010101.txt",5)

model = Wv(V,20)
ps = params(model)
opt = ADAGrad()

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

W_A1 = model.W
W1_A1 = model.W1

model = Wv(V,20)
ps = params(model)
data = create_batch("A01010102.txt",5)

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

W_A2 = model.W
W1_A2 = model.W1

model = Wv(V,20)
ps = params(model)
data = create_batch("B07020101.txt",5)

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

W_B = model.W
W1_B = model.W1

model = Wv(V,20)
ps = params(model)
data = create_batch("A01010101.txt",5)

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

W_A1bis = model.W
W1_A1bis = model.W1

simiA1 = similaire(W_A1+W1_A1',"AGC",10);
simiA1bis = similaire(W_A1bis+W1_A1bis',"AGC",10);
simiA2 = similaire(W_A2+W1_A2',"AGC",10);
simiB = similaire(W_B+W1_B',"AGC",10);

[simiA1 simiA1bis simiA2 simiB]

# azef

data = create_batch2(raw"Sequences\A01010101.txt",30)

model = Wv(V,20)
ps = Flux.params(model)
opt = ADAGrad()

loss = []

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

plot(loss)

W_A1 = model.W
W1_A1 = model.W1

println(eval(data))
println(eval2(data))
println(eval3(data,0.25,0.10))

model = Wv(V,20)
ps = params(model)
opt = ADAGrad()

loss = []

@time Flux.train!(cst, ps, Iterators.repeated(data,250), opt; cb = () -> update_loss!(loss,cst))

W_A1bis = model.W
W1_A1bis = model.W1

println(eval(data))
println(eval2(data))
println(eval3(data,0.25,0.10))

simiA1 = similaire(W_A1+W1_A1',"AGA",5);
simiA1bis = similaire(W_A1bis+W1_A1bis',"AGA",5);

[simiA1 simiA1bis]

W1 = W_A1+W1_A1'
W2 = W_A1bis+W1_A1bis'

e = evalsimi(W1,W2,5)
#= bar(e) =#
mean(e)

#= @time (eva_1 , eva_2 , eva_3, mean_simi) = tcontext(1:30; train = 200, type = 3 , sim = 6)

plot(eva_1, label = "eva_1")
plot(eva_2, label = "eva_2")
plot(eva_3, label = "eva_3")

m_simi = [ mean(mean_simi[i,:]) for i in 1:length(mean_simi[:,1])]

plot(m_simi, label = "m_simi") =#



#= sauvegarde("type3_30","tcontext(1:30; train = 200, type = 3 , sim = 6)",eva_1,eva_2,eva_3,mean_simi) =#