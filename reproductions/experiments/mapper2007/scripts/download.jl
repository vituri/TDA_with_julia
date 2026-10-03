using Downloads, SHA, TOML
root = dirname(@__DIR__)
data_dir = joinpath(root, "data")
provenance = TOML.parsefile(joinpath(data_dir, "provenance.toml"))
filehash(path) = bytes2hex(sha256(read(path)))
for source in provenance["sources"]
    path = joinpath(data_dir, source["file"])
    if !isfile(path)
        mkpath(dirname(path))
        temp = path * ".download"
        Downloads.download(source["url"], temp)
        filehash(temp) == source["sha256"] || error("Source checksum mismatch: $temp")
        mv(temp, path)
    end
    filehash(path) == source["sha256"] || error("Source checksum mismatch: $path")
end

# R is only needed to reconstruct the checked-in CSV from the upstream RData.
csv_path = joinpath(data_dir, "diabetes.csv")
if !isfile(csv_path) || "--rebuild-csv" in ARGS
    isnothing(Sys.which("Rscript")) && error("Rscript is required to rebuild diabetes.csv")
    code = "args <- commandArgs(trailingOnly=TRUE); load(args[1]); " *
           "write.csv(data.frame(patient_id=seq_len(nrow(diabetes)),diabetes), args[2], row.names=FALSE)"
    run(`Rscript --vanilla -e $code $(joinpath(data_dir, "raw", "diabetes.rda")) $csv_path`)
end
filehash(csv_path) == provenance["csv_sha256"] || error("CSV checksum mismatch: $csv_path")
println("Verified upstream files and the 145-patient CSV.")
