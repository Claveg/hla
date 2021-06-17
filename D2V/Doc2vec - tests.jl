using Plots
using Flux

include("W2V\\Donnees.jl")
include("D2V\\Doc2vec.jl")
include("D2V\\D2V_donnees.jl")
include("D2V\\imgt.jl")

V = 64
Vdoc = 10
hiddoc = 10
hidword = 10
c = 5

db2 = copy(db1[1:Vdoc])

@time (X,D,Y) = imgt(db2,c)

data = (X,D,Y)

model = Dv(V,Vdoc,hidword,hiddoc)
ps = Flux.params(model)

cost(x_w,x_d,y) = Flux.crossentropy( model(x_w,x_d), y)

c1 = cost(data...)
exp(-c1)

lost = []

opt = ADAGrad()

@time Flux.train!(cost, ps, Iterators.repeated(data,20), opt; cb = () -> update_loss!(lost,cost))

plot(lost)

(lost[1]-lost[end])/lost[1]

model2 = Dv(V,Vdoc,hidword,hiddoc)
ps2 = Flux.params(model2)
opt = ADAGrad()

lost = []

@time train!(cst2, ps2, Iterators.repeated(data,19), opt; cb = () -> update_loss2!(lost))
