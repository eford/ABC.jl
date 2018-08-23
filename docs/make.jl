using Documenter, ABC

makedocs(
    format = :html,
    sitename = "ABC.jl",
    modules = [ABC],
#    doctest = true, 
#    clean = false,
    pages = [
        "index.md", "page1.md",
    ],
    html_prettyurls = !("local" in ARGS)

)


deploydocs(
    repo   = "github.com/eford/ABC.jl.git",
    julia  = "0.6", 
    target = "build",
    deps   = nothing,
    make   = nothing
)

