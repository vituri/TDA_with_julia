using Pkg, SHA, Tar, TOML

const BOOK_ROOT = dirname(@__DIR__)
const PACKAGE_NAMES = [
    "MetricSpaces", "TDAPersistenceDiagrams", "TDARipserer",
    "TDAmapper", "TDAplots", "ToMATo",
]

function source_digest(root)
    files = String[]
    for (directory, _, names) in walkdir(root), name in names
        push!(files, relpath(joinpath(directory, name), root))
    end
    sort!(files; by = path -> replace(path, '\\' => '/'))
    digest = SHA.SHA2_256_CTX()
    for relative in files
        SHA.update!(digest, codeunits(replace(relative, '\\' => '/')))
        SHA.update!(digest, UInt8[0])
        SHA.update!(digest, sha256(read(joinpath(root, relative))))
    end
    bytes2hex(SHA.digest!(digest))
end

function check_sources(root, lock)
    for name in PACKAGE_NAMES
        package = joinpath(root, name * ".jl")
        isdir(package) || error("Missing frozen package: $package")
        expected = lock["packages"][name]["source_sha256"]
        source_digest(package) == expected ||
            error("Snapshot $package has changed. Move it aside before rerunning setup; your edits will not be overwritten.")
    end
end

function unpack_sources()
    directory = joinpath(BOOK_ROOT, "environment")
    lock = TOML.parsefile(joinpath(directory, "source-lock.toml"))
    archive = joinpath(directory, lock["archive"])
    bytes2hex(sha256(read(archive))) == lock["archive_sha256"] ||
        error("The package source archive does not match source-lock.toml.")
    destination = joinpath(BOOK_ROOT, ".book-packages")
    if isdir(destination)
        check_sources(destination, lock)
        return destination
    end
    staging = mktempdir(BOOK_ROOT; prefix = ".book-packages-staging-")
    try
        Tar.extract(archive, staging)
        check_sources(staging, lock)
        mv(staging, destination)
    finally
        isdir(staging) && rm(staging; recursive = true)
    end
    destination
end

VERSION >= v"1.12" || error("This book environment requires Julia 1.12 or later; it was checked with 1.12.5.")
all(arg -> arg in ("--local", "--no-kernel"), ARGS) ||
    error("Usage: julia scripts/setup.jl [--local] [--no-kernel]")
package_root = "--local" in ARGS ?
    get(ENV, "TDA_PACKAGE_ROOT", dirname(BOOK_ROOT)) : unpack_sources()
cd(BOOK_ROOT) do
    Pkg.activate(".")
    # Relative paths make the checked-in environment portable.
    Pkg.develop([PackageSpec(path = relpath(joinpath(package_root, name * ".jl"), BOOK_ROOT))
                 for name in PACKAGE_NAMES])
    Pkg.instantiate()
end
if !("--no-kernel" in ARGS)
    @eval using IJulia
    IJulia.installkernel("TDA book", "--project=" * BOOK_ROOT, "--startup-file=no";
                         specname = "tda-book")
end
println("Book environment ready. Run quarto render --cache-refresh from ", BOOK_ROOT)
