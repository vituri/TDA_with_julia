# Compare independent reduction modes on exactly the same published metric.
using TDARipserer, TDAPersistenceDiagrams, MetricSpaces, LinearAlgebra
experiment=@__DIR__
rows=[]
for signal in ("g1","g2")
    A=reduce(hcat,[parse.(Float64,split(row,','))
        for row in readlines(joinpath(experiment,"data",signal*"_published.csv"))[2:end]])
    X=A[7:11,:]
    for p in (2,3), algorithm in (:cohomology,:homology)
        d=ripserer(Rips(EuclideanSpace(X);threshold=Inf);dim_max=1,modulus=p,alg=algorithm,reps=false)[2]
        bar=d[argmax(persistence.(d))]
        push!(rows,(signal,p,string(algorithm),birth(bar),death(bar),persistence(bar)))
        println((signal,p,algorithm,birth(bar),death(bar),persistence(bar)))
        flush(stdout)
    end
end
open(joinpath(experiment,"results","algorithm_crosscheck.csv"),"w") do io
    println(io,"signal,modulus,algorithm,birth,death,max_persistence")
    foreach(row -> println(io,join(row,',')),rows)
end
for signal in ("g1","g2"),p in (2,3)
    group=filter(row->row[1]==signal&&row[2]==p,rows)
    @assert maximum(abs,collect(group[1][4:6]).-collect(group[2][4:6]))<1e-12
end
println("Both reduction modes agree on all four principal H1 bars.")
