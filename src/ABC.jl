module ABC

# package code goes here

using Distributions
using PDMats

export
  # types 
  abc_plan_type,
  abc_pmc_plan_type,
  abc_population_type,

  # util methods

  # abc alg methods
  generate_theta,
  init_abc,
  update_abc_pop,
  run_abc


import Base: mean, median, maximum, minimum, quantile, std, var, cov, cor
import Base: rand

# until added to distributions, use our own
include("GaussianMixtureModelCommonCovar.jl")

include("types.jl")
include("util.jl")
include("alg.jl")

end # module
