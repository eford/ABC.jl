function generate_theta(plan::abc_pmc_plan_type, sampler::Distribution, ss_true, epsilon::Float64; num_max_attempt = plan.num_max_attempt)
      @assert(epsilon>0.0)
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

# Generate an array of thetas (useful for parallelizing)
function generate_thetas(plan::abc_pmc_plan_type, sampler::Distribution, ss_true, epsilon::Float64, num_to_generate::Integer; num_max_attempt = plan.num_max_attempt)
    @assert(epsilon>0.0)
    ex_sample = rand(sampler)
    num_param = length(ex_sample)
    thetas = Array(eltype(ex_sample),num_param,num_to_generate)
    dists = Array(Float64, num_to_generate)  # Warning should make aware of distance return type
    attempts = Array(Int64,  num_to_generate)
    for i in 1:num_to_generate
      thetas[:,i], dists[i], attempts[i] = generate_theta(plan, sampler, ss_true, epsilon, num_max_attempt = num_max_attempt )
    end
    return (thetas, dists, attempts)
end

# Generate initial abc population from prior, aiming for d(ss,ss_true)<epsilon
function init_abc(plan::abc_pmc_plan_type, ss_true; in_parallel::Bool = false)
  if in_parallel
   nw = nworkers()
   @assert (nw > 1)
   @assert (plan.num_part > 2*nw)
   return init_abc_parallel_map(plan,ss_true)
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
  weights = fill(1.0//plan.num_part,plan.num_part)
  return abc_population_type(theta,weights,dist_theta)
end

### WARNING PARALLEL CODE UNTESTED
function init_abc_parallel_map(plan::abc_pmc_plan_type, ss_true)
  num_param = length(Distributions.rand(plan.prior))
  num_draw = plan.num_part
  zip(theta_gen, dist_theta, attempts) = pmap(x->generate_thetas(plan, plan.prior, ss_true, plan.epsilon_init), [1:plan.num_part])
  for i in 1:plan.num_part
      theta[:,i]  = theta_gen[i]
  end

  weights = fill(1.0//plan.num_part,plan.num_part)
  return abc_population_type(theta,weights,dist_theta)
end

function init_abc_parallel_custom(plan::abc_pmc_plan_type, ss_true)
  num_param = length(Distributions.rand(plan.prior))
  num_draw = plan.num_part
  nw = nworkers()

  job = Array(RemoteRef, nw)
  a = [1:ifloor(num_draw/nw):num_draw][1:nw]
  b = Array(Int64,nw)
  for i in 1:nw-1
     b[i] = a[i+1]-1
  end
  b[end] = num_draw

  for j in 1:nw
    #println("# Starting jobs ",j," with entries ", a[j], " to ", b[j])
    # Draw initial set of theta's from prior (either with dist<epsilon or best of num_max_attempts)
    job[j] = @spawn generate_thetas(plan, plan.prior, ss_true, plan.epsilon_init, b[j]-a[j]+1)  # TODO NEED TO MAKE THIS RETURN ARRAYS for i in a[j]:b[j] ]
  end

  # Allocate arrays for results on master
  theta = Array(Float64,(num_param,plan.num_part))
  dist_theta = Array(Float64, plan.num_part)
  attempts = Array(Int64, plan.num_part)
  # Retreive results
  for j in 1:nw
    theta[:,a[j]:b[j]], dist_theta[a[j]:b[j]], attempts[a[j]:b[j]] = fetch(job[j])
  end
  weights = fill(1.0//plan.num_part,plan.num_part)
  return abc_population_type(theta,weights,dist_theta)
end


# Update the abc population once
function update_abc_pop_parallel(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type, epsilon::Float64;
                        attempts::Array{Int64,1} = zeros(Int64,plan.num_part))
  new_pop = copy(pop)
  # define sampler to be used
  theta_mean = sum(pop.theta.*pop.weights') # weighted mean for parameters
  tau = plan.tau_factor*cov_weighted(pop.theta'.-theta_mean,pop.weights)  # scaled, weighted covar for parameters
  sampler = GaussianMixtureModelCommonCovar(pop.theta,pop.weights,tau)

  zip(theta_star, dist_theta_star, attempts) = pmap(x->generate_theta(plan, sampler, ss_true, epsilon), [1:plan.num_part])
     for i in 1:plan.num_part
       #theta_star, dist_theta_star, attempts[i] = generate_theta(plan, sampler, ss_true, epsilon)
       # if dist_theta_star < pop.dist[i] # replace theta with new set of parameters and update weight
       if dist_theta_star[i] < epsilon # replace theta with new set of parameters and update weight
         @inbounds new_pop.dist[i] = dist_theta_star[i]
         @inbounds new_pop.theta[:,i] = theta_star[i]
         prior_pdf = Distributions.pdf(plan.prior,theta_star[i])
         # sampler_pdf calculation must match distribution used to update particle
         sampler_pdf = pdf(sampler, theta_star[i] )
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
   return new_pop
end


# Update the abc population once
function update_abc_pop(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type, epsilon::Float64;
                        attempts::Array{Int64,1} = zeros(Int64,plan.num_part))
  new_pop = copy(pop)
  # define sampler to be used
  theta_mean = sum(pop.theta.*pop.weights') # weighted mean for parameters
  tau = plan.tau_factor*cov_weighted(pop.theta'.-theta_mean,pop.weights)  # scaled, weighted covar for parameters
  sampler = GaussianMixtureModelCommonCovar(pop.theta,pop.weights,tau)
     for i in 1:plan.num_part
       theta_star, dist_theta_star, attempts[i] = generate_theta(plan, sampler, ss_true, epsilon)
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
   return new_pop
end

# run the ABC algorithm matching to summary statistics ss_true, starting from an initial population (e.g., output of previosu call)
function run_abc(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type; verbose::Bool = false, print_every::Integer=1, in_parallel::Bool = false )
  attempts = zeros(Int64,plan.num_part)
  # Set initial epsilon tolerance based on current population
  epsilon = quantile(pop.dist,plan.init_epsilon_quantile)
  for t in 1:plan.num_max_times
    local new_pop
    if in_parallel
      new_pop = update_abc_pop_parallel(plan, ss_true, pop, epsilon, attempts=attempts)
    else
      new_pop = update_abc_pop(plan, ss_true, pop, epsilon, attempts=attempts)
    end
    pop = copy(new_pop)
    if verbose && (t%print_every == 0)
       println("# t= ",t, " eps= ",epsilon, " med(d)= ",median(pop.dist), " attempts= ",median(attempts), " ",maximum(attempts), " reps= ", sum(pop.repeats), " ess= ",ess(pop.weights,pop.repeats)) #," mean(theta)= ",mean(pop.theta,2) )#) #, " tau= ",diag(tau) ) #
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
    if maximum(attempts)<0.75*plan.num_max_attempt
      epsilon = minimum([maximum(pop.dist),epsilon * plan.epsilon_reduction_factor])
    end
  end # t / num_times
  #println("mean(theta) = ",[ sum(pop.theta[i,:])/size(pop.theta,2) for i in 1:size(pop.theta,1) ])
  return pop
end

# run the ABC algorithm matching to summary statistics ss_true
function run_abc(plan::abc_pmc_plan_type, ss_true; verbose::Bool = false, print_every::Integer=1, in_parallel::Bool = false )                                          # Initialize population, drawing from prior
  pop::abc_population_type = init_abc(plan,ss_true, in_parallel=in_parallel)
  #println("pop_init: ",pop)
  run_abc(plan, ss_true, pop; verbose=verbose, print_every=print_every, in_parallel=in_parallel )  # Initialize population, drawing from prior
end


