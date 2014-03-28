module ABC

# package code goes here

using Distributions
# until added to distributions, use our own
include("GaussianMixtureModelCommonCovar.jl")

include("types.jl")
include("utils.jl")
include("alg.jl")

end # module
