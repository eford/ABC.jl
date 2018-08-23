using Documenter, ABC

makedocs(
    modules = [ABC],
    doctest = true, 
    clean = false,
    format = :html,
    sitename = "ABC.jl",
    pages = [
        "index.md",
    ],
    html_prettyurls = !("local" in ARGS)

)


deploydocs(
    repo   = "github.com/eford/ABC.jl.git",
    julia  = "0.6.0", 
    target = "build",
    deps   = nothing,
    make   = nothing
)

