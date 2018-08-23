var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "ABC.jl:  Approximate Bayesian Computing for Julia",
    "title": "ABC.jl:  Approximate Bayesian Computing for Julia",
    "category": "page",
    "text": ""
},

{
    "location": "#ABC.jl:-Approximate-Bayesian-Computing-for-Julia-1",
    "page": "ABC.jl:  Approximate Bayesian Computing for Julia",
    "title": "ABC.jl:  Approximate Bayesian Computing for Julia",
    "category": "section",
    "text": "This package provides for Approximate Bayesian Computing (ABC) via sequential importance sampling in Julia.   Currently, it implements a single algorithm, ABC-PMC based on Beaumont et al. 2002 via abc_pmc_plan_type. However, it does include several extra features that will eventually be documented. "
},

{
    "location": "#Getting-Started-1",
    "page": "ABC.jl:  Approximate Bayesian Computing for Julia",
    "title": "Getting Started",
    "category": "section",
    "text": "Add the ABC package (update once become registered)\nusing ABC\ninclude(\"tests/runtests.jl\")\nSee test1.jl for a demo of how to use ABC"
},

{
    "location": "#API-1",
    "page": "ABC.jl:  Approximate Bayesian Computing for Julia",
    "title": "API",
    "category": "section",
    "text": "Modules = [ABC]\nPrivate = false\nOrder   = [:module, :function, :type]"
},

{
    "location": "#Index-1",
    "page": "ABC.jl:  Approximate Bayesian Computing for Julia",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "page1/#ABC",
    "page": "-",
    "title": "ABC",
    "category": "module",
    "text": "ABC Module providing     types (abc_pmc_plan_type, abc_population_type) and methods (generate_theta, init_abc, update_abc_pop, run_abc) for using Approximate Bayesian Computing\n\n\n\n"
},

{
    "location": "page1/#ABC.abc_population_type-Tuple{Integer,Integer}",
    "page": "-",
    "title": "ABC.abc_population_type",
    "category": "method",
    "text": "abc_population_type(num_param::Integer, num_particles::Integer; accept_log::abc_log_type = abc_log_type(), reject_log::abc_log_type = abc_log_type(), repeats::Array{Int64,1} = zeros(Int64,num_particles) )\n\nnum_param:  Number of model parameters for generating simulated data num_particles: Number of particles for sequential importance sampler\n\nOptional parameters: accept_log: Log of accepted parameters/summary statistics/distances reject_log: Log of rejected parameters/summary statistics/distances repeats:    Array indicating which particles have been repeated from previous generation\n\n\n\n"
},

{
    "location": "page1/#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": "ABCabc_pmc_plan_type(gd::Function,css::Function,cd::Function,p)ABC.abc_population_type(num_param::Integer, num_particles::Integer)"
},

]}
