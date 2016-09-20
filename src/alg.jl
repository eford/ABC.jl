#using PyCall
#unshift!(PyVector(pyimport("sys")["path"]), "")
#@pyimport nearest_correlation

#include("alg_parallel_custom.jl")   # Demo of using @spawn and fetch to reduce memory usage

function generate_theta(plan::abc_pmc_plan_type, sampler::Distribution, ss_true, epsilon::Float64; num_max_attempt = plan.num_max_attempt)
      @assert(epsilon>=0.0)
      dist_best = Inf
      local theta_best
      attempts = num_max_attempt
      for a in 1:num_max_attempt
         theta_star = rand(sampler)
         plan.normalize(theta_star)
         if(!plan.is_valid(theta_star)) continue end
         data_star = plan.gen_data(theta_star)
         ss_star = plan.calc_summary_stats(data_star)
         dist_star = plan.calc_dist(ss_true,ss_star)
         if dist_star < dist_best
            dist_best = dist_star
            theta_best = copy(theta_star)
         end
         if(dist_best < epsilon)
            #println("Current distance: ", dist_best, " / Current rate: ", exp(theta_best[1]))
            attempts = a
            break
         end
      end
      if dist_best == Inf
        error("# Failed to generate any acceptable thetas.")
      end
      # println("gen_theta: d= ",dist_best, " a= ",attempts, " theta= ", theta_best)
      return (theta_best, dist_best, attempts)
end

# Generate initial abc population from prior, aiming for d(ss,ss_true)<epsilon
function init_abc(plan::abc_pmc_plan_type, ss_true; in_parallel::Bool = plan.in_parallel)
  if in_parallel
   nw = nworkers()
   @assert (nw > 1)
   #@assert (plan.num_part > 2*nw)  # Not really required, but seems more likely to be a mistake
   #return init_abc_parallel_map(plan,ss_true)
   return init_abc_distributed_map(plan,ss_true)
  else
   return init_abc_serial(plan,ss_true)
  end
end

function init_abc_serial(plan::abc_pmc_plan_type, ss_true)
  num_param = length(Distributions.rand(plan.prior))
  # Allocate arrays
  theta = Array(Float64,(num_param,plan.num_part))
  dist_theta = Array(Float64,plan.num_part)
  attempts = zeros(plan.num_part)
  # Draw initial set of theta's from prior (either with dist<epsilon or best of num_max_attempts)

  for i in 1:plan.num_part
      theta[:,i], dist_theta[i], attempts[i] = generate_theta(plan, plan.prior, ss_true, plan.epsilon_init)
  end
  weights = fill(1.0/plan.num_part,plan.num_part)
  return abc_population_type(theta,weights,dist_theta)
end

function init_abc_distributed_map(plan::abc_pmc_plan_type, ss_true)
  #num_param = length(Distributions.rand(plan.prior))
  num_param = length(plan.prior)
  darr_silly = dzeros(plan.num_part)
  map_results = map(x->generate_theta(plan, plan.prior, ss_true, plan.epsilon_init), darr_silly )
  @assert( length(map_results) >= 1)
  @assert( length(map_results[1]) >= 1)
  #num_param = length(map_results[1][1])
  theta = Array(Float64,(num_param,plan.num_part))
  dist_theta = Array(Float64,plan.num_part)
  attempts = Array(Int64,plan.num_part)
  for i in 1:plan.num_part
      theta[:,i]  = map_results[i][1]
      dist_theta[i]  = map_results[i][2]
      attempts[i]  = map_results[i][3]
  end

  weights = fill(1.0/plan.num_part,plan.num_part)
  return abc_population_type(theta,weights,dist_theta)
end

function init_abc_parallel_map(plan::abc_pmc_plan_type, ss_true)
  #num_param = length(Distributions.rand(plan.prior))
  pmap_results = pmap(x->generate_theta(plan, plan.prior, ss_true, plan.epsilon_init), collect(1:plan.num_part) )
  @assert( length(pmap_results) >= 1)
  @assert( length(pmap_results[1]) >= 1)
  num_param = length(pmap_results[1][1])
  theta = Array(Float64,(num_param,plan.num_part))
  dist_theta = Array(Float64,plan.num_part)
  attempts = Array(Int64,plan.num_part)
  for i in 1:plan.num_part
      theta[:,i]  = pmap_results[i][1]
      dist_theta[i]  = pmap_results[i][2]
      attempts[i]  = pmap_results[i][3]
  end

  weights = fill(1.0/plan.num_part,plan.num_part)
  return abc_population_type(theta,weights,dist_theta)
end

function make_proposal_dist_gaussian_full_covar(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false)
  theta_mean = sum(pop.theta.*pop.weights') # weighted mean for parameters
  rawtau = cov_weighted(pop.theta'.-theta_mean,pop.weights)  # scaled, weighted covar for parameters
  tau = tau_factor*make_matrix_pd(rawtau)
  if verbose
    println("theta_mean = ", theta_mean)
    println("pop.theta = ", pop.theta)
    println("pop.weights = ", pop.weights)
    println("tau = ", tau)
  end
  covar = PDMat(tau)
  sampler = GaussianMixtureModelCommonCovar(pop.theta,pop.weights,covar)
end

function make_proposal_dist_gaussian_diag_covar(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false)
  theta_mean = sum(pop.theta.*pop.weights') # weighted mean for parameters
  tau = tau_factor*var_weighted(pop.theta'.-theta_mean,pop.weights)  # scaled, weighted covar for parameters
  if verbose
    println("theta_mean = ", theta_mean)
    println("pop.theta = ", pop.theta)
    println("pop.weights = ", pop.weights)
    println("tau = ", tau)
  end
  covar = PDiagMat(tau)
  sampler = GaussianMixtureModelCommonCovar(pop.theta,pop.weights,covar)
end

# Update the abc population once
function update_abc_pop_parallel_pmap(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type, epsilon::Float64;
                        attempts::Array{Int64,1} = zeros(Int64,plan.num_part))
  new_pop = copy(pop)
  sampler = plan.make_proposal_dist(pop, plan.tau_factor)

  pmap_results = pmap(x->generate_theta(plan, sampler, ss_true, epsilon), collect(1:plan.num_part))
     for i in 1:plan.num_part
       #theta_star, dist_theta_star, attempts[i] = generate_theta(plan, sampler, ss_true, epsilon)
       # if dist_theta_star < pop.dist[i] # replace theta with new set of parameters and update weight
       theta_star = pmap_results[i][1]
       dist_theta_star =  pmap_results[i][2]
       if dist_theta_star < epsilon # replace theta with new set of parameters and update weight
         @inbounds new_pop.theta[:,i] = theta_star
         @inbounds new_pop.dist[i] = dist_theta_star
         prior_pdf = Distributions.pdf(plan.prior,theta_star)
         # sampler_pdf calculation must match distribution used to update particle
         sampler_pdf = pdf(sampler, theta_star )
         @inbounds new_pop.weights[i] = prior_pdf/sampler_pdf
         @inbounds new_pop.repeats[i] = 0
       else  # failed to generate a closer set of parameters, so...
         # ... generate new data set with existing parameters
         #new_data = plan.gen_data(theta_star)
         #new_ss = plan.calc_summary_stats(new_data)
         #@inbounds new_pop.dist[i] = plan.calc_dist(ss_true,new_ss)
         # ... just keep last value for this time, and mark it as a repeat
         @inbounds new_pop.theta[:,i] = pop.theta[:,i]
         @inbounds new_pop.dist[i] = pop.dist[i]
         @inbounds new_pop.weights[i] = pop.weights[i]
         @inbounds new_pop.repeats[i] += 1
       end
     end # i / num_parts
   new_pop.weights ./= sum(new_pop.weights)
   return new_pop
end

function update_abc_pop_parallel_darray(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type, epsilon::Float64;
                        attempts::Array{Int64,1} = zeros(Int64,plan.num_part))
  new_pop = copy(pop)
  sampler = plan.make_proposal_dist(pop, plan.tau_factor)

  darr_silly = dzeros(plan.num_part)
  dmap_results = map(x->generate_theta(plan, sampler, ss_true, epsilon), darr_silly)
  map_results = collect(dmap_results)
     for i in 1:plan.num_part
       #theta_star, dist_theta_star, attempts[i] = generate_theta(plan, sampler, ss_true, epsilon)
       # if dist_theta_star < pop.dist[i] # replace theta with new set of parameters and update weight
       theta_star = map_results[i][1]
       dist_theta_star =  map_results[i][2]
       if dist_theta_star < epsilon # replace theta with new set of parameters and update weight
         @inbounds new_pop.theta[:,i] = theta_star
         @inbounds new_pop.dist[i] = dist_theta_star
         prior_pdf = Distributions.pdf(plan.prior,theta_star)
         # sampler_pdf calculation must match distribution used to update particle
         sampler_pdf = pdf(sampler, theta_star )
         @inbounds new_pop.weights[i] = prior_pdf/sampler_pdf
         @inbounds new_pop.repeats[i] = 0
       else  # failed to generate a closer set of parameters, so...
         # ... generate new data set with existing parameters
         #new_data = plan.gen_data(theta_star)
         #new_ss = plan.calc_summary_stats(new_data)
         #@inbounds new_pop.dist[i] = plan.calc_dist(ss_true,new_ss)
         # ... just keep last value for this time, and mark it as a repeat
         @inbounds new_pop.theta[:,i] = pop.theta[:,i]
         @inbounds new_pop.dist[i] = pop.dist[i]
         @inbounds new_pop.weights[i] = pop.weights[i]
         @inbounds new_pop.repeats[i] += 1
       end
     end # i / num_parts
   new_pop.weights ./= sum(new_pop.weights)
   return new_pop
end


# Update the abc population once
function update_abc_pop_serial(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type, epsilon::Float64;
                        attempts::Array{Int64,1} = zeros(Int64,plan.num_part))
  new_pop = copy(pop)
  sampler = plan.make_proposal_dist(pop, plan.tau_factor)

     for i in 1:plan.num_part
       theta_star, dist_theta_star, attempts[i] = generate_theta(plan, sampler, ss_true, epsilon)
      # theta_star, dist_theta_star, attempts[i] = try
      #   generate_theta(plan, sampler, ss_true, epsilon)
      # catch
      #   tau = nearest_correlation.nearcorr(tau)
      #   sampler = GaussianMixtureModelCommonCovarDiagonal(pop.theta,pop.weights,tau)
      #   generate_theta(plan, sampler, ss_true, epsilon)
      # end
       # if dist_theta_star < pop.dist[i] # replace theta with new set of parameters and update weight
       if dist_theta_star < epsilon # replace theta with new set of parameters and update weight
         @inbounds new_pop.dist[i] = dist_theta_star
         @inbounds new_pop.theta[:,i] = theta_star
         prior_pdf = Distributions.pdf(plan.prior,theta_star)
         # sampler_pdf calculation must match distribution used to update particle
         sampler_pdf = pdf(sampler, theta_star )
         @inbounds new_pop.weights[i] = prior_pdf/sampler_pdf
         @inbounds new_pop.repeats[i] = 0
       else  # failed to generate a closer set of parameters, so...
         # ... generate new data set with existing parameters
         #new_data = plan.gen_data(theta_star)
         #new_ss = plan.calc_summary_stats(new_data)
         #@inbounds new_pop.dist[i] = plan.calc_dist(ss_true,new_ss)
         # ... just keep last value for this time, and mark it as a repeat
         @inbounds new_pop.dist[i] = pop.dist[i]
         @inbounds new_pop.theta[:,i] = pop.theta[:,i]
         @inbounds new_pop.weights[i] = pop.weights[i]
         @inbounds new_pop.repeats[i] += 1
       end
     end # i / num_parts
   new_pop.weights ./= sum(new_pop.weights)
   #println("New pop weights = ", new_pop.weights)
   return new_pop
end

# run the ABC algorithm matching to summary statistics ss_true, starting from an initial population (e.g., output of previosu call)
function run_abc(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type; verbose::Bool = false, print_every::Integer=1, in_parallel::Bool = plan.in_parallel )
  attempts = zeros(Int64,plan.num_part)
  # Set initial epsilon tolerance based on current population
  epsilon = quantile(pop.dist,plan.init_epsilon_quantile)
  eps_diff_count = 0
  mean_arr = []
  std_arr = []
  eps_arr = []
  for t in 1:plan.num_max_times
    local new_pop
    if in_parallel
      new_pop = update_abc_pop_parallel_darray(plan, ss_true, pop, epsilon, attempts=attempts)
    else
      new_pop = update_abc_pop_serial(plan, ss_true, pop, epsilon, attempts=attempts)
    end
    pop = copy(new_pop)
    push!(mean_arr, mean(exp(pop.theta), 2)[1])
    push!(std_arr, std(exp(pop.theta), 2)[1])
    push!(eps_arr, epsilon)
    if verbose && (t%print_every == 0)
       println("# t= ",t, " eps= ",epsilon, " med(d)= ",median(pop.dist), " attempts= ",median(attempts), " ",maximum(attempts), " reps= ", sum(pop.repeats), " ess= ",ess(pop.weights,pop.repeats)) #," mean(theta)= ",mean(pop.theta,2) )#) #, " tau= ",diag(tau) ) #
       println("Mean(theta)= ", mean(exp(pop.theta), 2), " Stand. Dev.(theta)= ", std(exp(pop.theta), 2))
       # println("# t= ",t, " eps= ",epsilon, " med(d)= ",median(pop.dist), " max(d)= ", maximum(pop.dist), " med(attempts)= ",median(attempts), " max(a)= ",maximum(attempts), " reps= ", sum(pop.repeats), " ess= ",ess(pop.weights,pop.repeats)) #," mean(theta)= ",mean(pop.theta,2) )#) #, " tau= ",diag(tau) ) #
    end
    #if epsilon < plan.target_epsilon  # stop once acheive goal
    if maximum(pop.dist) < plan.target_epsilon  # stop once acheive goal
       println("# Reached ",epsilon," after ", t, " generations.")
       break
    end
    if sum(pop.repeats)>plan.num_part
      println("# Halting due to ", sum(pop.repeats), " repeats.")
      break
    end
    eps_old = epsilon
    if maximum(attempts)<0.75*plan.num_max_attempt
      epsilon = minimum([maximum(pop.dist),epsilon * plan.epsilon_reduction_factor])
    end
    if ((abs(eps_old-epsilon)/epsilon) < 1.0e-5)
      eps_diff_count += 1
    end
    if eps_diff_count > 4
      println("# Halting due to epsilon not improving significantly for 5 consecutive generations.")
      break
    end
  end # t / num_times
  #println("mean(theta) = ",[ sum(pop.theta[i,:])/size(pop.theta,2) for i in 1:size(pop.theta,1) ])
  println("Epsilon history = ", eps_arr)
  println("Mean history = ", mean_arr)
  println("Std Dev. history = ", std_arr)
  return pop
end

# run the ABC algorithm matching to summary statistics ss_true
function run_abc(plan::abc_pmc_plan_type, ss_true; verbose::Bool = false, print_every::Integer=1, in_parallel::Bool =  plan.in_parallel )                                          # Initialize population, drawing from prior
  pop::abc_population_type = init_abc(plan,ss_true, in_parallel=in_parallel)
  #println("pop_init: ",pop)
  run_abc(plan, ss_true, pop; verbose=verbose, print_every=print_every, in_parallel=in_parallel )  # Initialize population, drawing from prior
end


