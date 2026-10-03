using TDARipserer, TDAPersistenceDiagrams, MetricSpaces, Random, Statistics
rng=MersenneTwister(2011)
u=rand(rng,200)
A=hcat(cos.(2π.*u),sin.(2π.*u)).+0.4.*rand(rng,200,2)
X=EuclideanSpace(permutedims(A))
filt=Rips(X;threshold=.5)
dgm=ripserer(filt;dim_max=1,modulus=47,reps=true)[2]
for δ in (.14,.4)
 bars=sort(filter(i->birth(i)<δ<death(i),dgm);by=i->min(death(i),.5)-birth(i),rev=true)
 println("circle delta=",δ," active=",length(bars)," bars=",[(birth(i),death(i),length(i.representative)) for i in bars])
end
rng=MersenneTwister(2011)
u=rand(rng,400);v=rand(rng,400)
A=hcat((2 .+cos.(2π.*v)).*cos.(2π.*u),(2 .+cos.(2π.*v)).*sin.(2π.*u),sin.(2π.*v)) .+ 0.2 .*rand(rng,400,3)
X=EuclideanSpace(permutedims(A));filt=Rips(X;threshold=sqrt(3));dgm=ripserer(filt;dim_max=1,modulus=47,reps=true)[2]
for δ in (1.4,1.6)
 bars=sort(filter(i->birth(i)<δ<death(i),dgm);by=i->min(death(i),sqrt(3))-birth(i),rev=true)
 println("torus delta=",δ," active=",length(bars)," bars=",[(birth(i),death(i),length(i.representative)) for i in bars[1:min(5,length(bars))]])
end
