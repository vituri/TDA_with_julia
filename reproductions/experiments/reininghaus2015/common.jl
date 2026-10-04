using TDAPersistenceDiagrams, DelimitedFiles, LinearAlgebra, SHA, Statistics, Serialization
const ROOT = @__DIR__
const NORMALIZED = "--normalized" in ARGS
const RESULTS = NORMALIZED ? joinpath(ROOT,"results","normalized") : joinpath(ROOT,"results")
const SOURCE_SHA256 = "80cfce72ee053ffac2ee82e7358118bc43907930262b795ef443460a3abcd042"
const SOURCE_COMMIT = "3e217d151d09e5eed213f927960986854331c148"
const SOURCE_URL = "https://raw.githubusercontent.com/lucho8908/adaptive_template_systems/$SOURCE_COMMIT/Examples/Shapes/Uli_data/Uli_data.csv"
const SIGMAS = 2.0 .^ collect(-12:2:16)
const COSTS = 2.0 .^ collect(-5:2:15)
const SEED = 20150315
const PAPER_CLASSIFICATION = [94.7,99.3,96.3,97.3,96.3,93.7,88.0,88.3,88.0,91.0]
const PAPER_SD = [5.1,0.9,2.2,1.9,2.5,3.2,4.5,6.0,5.8,4.0]
const PAPER_RETRIEVAL = [88.7,94.7,91.3,93.0,92.3,77.3,80.0,80.7,83.0,69.3]

function load_data()
    file = joinpath(ROOT,"data","Uli_data.csv")
    bytes = read(file)
    bytes2hex(sha256(bytes)) == SOURCE_SHA256 || error("source hash mismatch")
    rows = readdlm(IOBuffer(bytes), ',', Float64; header=true)[1]
    # The redistributed CSV calls shape ID `freq` and HKS index `trial`.
    pairs = [[Tuple{Float64,Float64}[] for i in 1:300] for t in 1:10]
    diagonal = zeros(Int,10)
    essential = zeros(Int,10)
    for r in eachrow(rows)
        shape, t, d = Int(r[1])+1, Int(r[2]), Int(r[3])
        1 <= shape <= 300 && 1 <= t <= 10 || error("unexpected record ID")
        if d < 0
            essential[t] += 1
        elseif d == 1
            r[5] >= r[4] || error("negative lifetime in source")
            if r[5] == r[4]
                diagonal[t] += 1 # CSV rounding; exactly zero in the PSS kernel.
            else
                push!(pairs[t][shape],(r[4],r[5]))
            end
        end
    end
    diagrams = [[PersistenceDiagram(p;dim=1) for p in ps] for ps in pairs]
    labels = repeat(collect(1:15),inner=20)
    return (;rows,diagrams,labels,diagonal,essential)
end

function pss_matrix(diagrams,sigma)
    # Thread calls to the library's kernel; no approximation or point truncation.
    k = PersistenceScaleSpaceKernel(;sigma)
    n = length(diagrams)
    G = zeros(n,n)
    Threads.@threads :dynamic for j in 1:n
        for i in 1:j
            G[i,j] = G[j,i] = k(diagrams[i],diagrams[j])
        end
    end
    return G
end

function nearest_neighbors(K,labels)
    n = length(labels)
    predicted = zeros(Int,n)
    distance = zeros(n)
    for i in 1:n
        # Squared RKHS distance; never include the query itself.
        distances = [j==i ? Inf : max(0.0,K[i,i]+K[j,j]-2K[i,j]) for j in 1:n]
        j = argmin(distances) # smallest shape ID resolves exact ties.
        predicted[i]=j
        distance[i]=distances[j]
    end
    accuracy=100mean(labels[predicted].==labels)
    return (;accuracy,predicted,distance)
end

function normalize_gram(K)
    # Unit norm for nonzero embeddings; zero diagrams retain zero rows. This
    # extension avoids the 0/0 of the authors' later MATLAB utility.
    z=sqrt.(max.(diag(K),0.0))
    G=zeros(size(K))
    for j in axes(K,2),i in axes(K,1)
        z[i]>0&&z[j]>0&&(G[i,j]=K[i,j]/(z[i]*z[j]))
    end
    return G
end
