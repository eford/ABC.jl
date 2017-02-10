function draw_theta_valid(plan::abc_pmc_plan_type, sampler::Distribution; num_max_attempt::Integer = plan.num_max_attempt)
  #local theta_star
  for a in 1:num_max_attempt
         theta_star = rand(sampler)
         if issubtype(typeof(theta_star),Real)   # in case return a scalar, make into array
            theta_star = [theta_star]
         end
         plan.normalize(theta_star)
         if(plan.is_valid(theta_star)) return theta_star end
  end
  error("# Failed to draw valid theta after ", num_max_attempt, " attemps.")
end

function generate_theta(plan::abc_pmc_plan_type, sampler::Distribution, ss_true, epsilon::Float64; num_max_attempt::Integer = plan.num_max_attempt)
      @assert(epsilon>0.0)
      dist_best = Inf
      local theta_best
      #summary_stats_log = abc_log_type()
      #push!(summary_stats_log.generation_starts_at, 1)
      accept_log = abc_log_type()
      reject_log = abc_log_type()
      push!(accept_log.generation_starts_at, 1)
      push!(reject_log.generation_starts_at, 1)
      attempts = 0
      accepts_emulator = 0
      indecisive_emulator = 0
      rejects_emulator = 0
      accepts_full_model = 0
      rejects_full_model = 0
      all_attempts = num_max_attempt
      for a in 1:num_max_attempt
         theta_star = rand(sampler)
         if issubtype(typeof(theta_star),Real)   # in case return a scalar, make into array
            theta_star = [theta_star]
         end
         plan.normalize(theta_star)
         if(!plan.is_valid(theta_star)) continue end
         attempts += 1
         run_full_model = true
         if plan.use_emulator
            ss_star = plan.emulator(theta_star)
            dist_star = plan.calc_dist(ss_true,ss_star)
            is_best_dist = dist_star < dist_best

            emu_decission = emu_accept_reject_run_full_model(ss_true,ss_star,epsilon, prob_accept_crit = 0.999, prob_reject_crit = 0.001)
               #println("# Emulator ", emu_decission, " theta =",theta_star, " dist=",dist_star," ss=",ss_star)
            if emu_decission == 1
               accept_star = true
               run_full_model = true
               println("# Emulator accepted theta =",theta_star, " dist=",dist_star," ss=",ss_star)
               accepts_emulator += 1
            elseif emu_decission == -1
               accept_star = false
               run_full_model = false
               #println("# Emulated rejected theta =",theta_star, " dist=",dist_star," ss=",ss_star)
               rejects_emulator += 1
            elseif emu_decission == 0
               indecisive_emulator += 1
               #=
               #run_full_model = true
               data_star = plan.gen_data(theta_star)
               ss_star = plan.calc_summary_stats(data_star)
               dist_star = plan.calc_dist(ss_true,ss_star)
               is_best_dist = dist_star < dist_best
               accept_star = dist_star < epsilon
               =#
            else
             error("Emulator decission wasn't a valid option: ",emu_decission," ss=",ss_star, " dist=",dist_star)
            end
         end
         if run_full_model
            data_star = plan.gen_data(theta_star)
            ss_star = plan.calc_summary_stats(data_star)
            dist_star = plan.calc_dist(ss_true,ss_star)
            is_best_dist = dist_star < dist_best
            accept_star = dist_star < epsilon
            if accept_star
               accepts_full_model += 1
      else

      rejects_full_model += 1
      end
#if plan.use_emulator
            #   println("# theta =",theta_star," ss=",ss_star, " dist=",dist_star)
            #end
         end

         if is_best_dist
            dist_best = dist_star
            theta_best = copy(theta_star)
         end

         if accept_star
            #if run_full_model
               #println("# Accepting full model theta =",theta_star," d=",dist_star)#, "   ss=",ss_star)
               push_to_abc_log!(accept_log,plan,theta_star,ss_star,dist_star,run_full_model)
            #end
            all_attempts = a
            break
         else
            if run_full_model
               #println("# Rejecting full model theta =",theta_star," d=",dist_star)#, "   ss=",ss_star)
               push_to_abc_log!(reject_log,plan,theta_star,ss_star,dist_star,run_full_model)
            end
         end
      end
      if dist_best == Inf
        error("# Failed to generate any acceptable thetas.")
      end
      # println("gen_theta: d= ",dist_best, " num_valid_attempts= ",attempts, " num_all_attempts= ", all_attempts, " theta= ", theta_best)
      #return (theta_best, dist_best, attempts, summary_stats_log)
      #println("# attempts: ", attempts, " Emulator: ",accepts_emulator, " ", rejects_emulator, " ", indecisive_emulator, " Full Model: ", accepts_full_model,  " ",rejects_full_model)
      return (theta_best, dist_best, attempts, accept_log, reject_log)
end

function generate_abc_sample(plan::abc_plan_type, pop::abc_population_type, ss_true, epsilon::Real; num_plot::Integer = 100)
  num_param = size(pop.theta,1)
  theta_plot = Array(Float64,(num_param, num_plot))
  dist_plot = Array(Float64,num_plot)
  sampler = plan.make_proposal_dist(pop, plan.tau_factor)
  for i in 1:num_plot
    theta_plot[:,i], dist_plot[i], attempts_plot, accept_log_plot, reject_log_plot = ABC.generate_theta(plan, sampler, ss_true, epsilon)
  end
  return theta_plot, dist_plot
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
  #summary_stat_logs = Array(abc_log_type,plan.num_part)
  #summary_stat_log_combo = abc_log_type()
  accept_log_combo = abc_log_type()
  reject_log_combo = abc_log_type()
  # Draw initial set of theta's from prior (either with dist<epsilon or best of num_max_attempts)

  if plan.save_params || plan.save_summary_stats || plan.save_distances
     push!(accept_log_combo.generation_starts_at,length(accept_log_combo.dist)+1)
     push!(reject_log_combo.generation_starts_at,length(reject_log_combo.dist)+1)
  end
  for i in 1:plan.num_part
      theta[:,i], dist_theta[i], attempts[i], accept_log, reject_log = generate_theta(plan, plan.prior, ss_true, plan.epsilon_init)
      append_to_abc_log!(accept_log_combo,plan,accept_log.theta,accept_log.ss,accept_log.dist,accept_log.full_model)
      append_to_abc_log!(reject_log_combo,plan,reject_log.theta,reject_log.ss,reject_log.dist,reject_log.full_model)
      #=
      if plan.save_params
         append!(accept_log_combo.theta, accept_log.theta)
         append!(reject_log_combo.theta, reject_log.theta)
      end
      if plan.save_summary_stats
         append!(accept_log_combo.ss, accept_log.ss)
         append!(reject_log_combo.ss, reject_log.ss)
      end
      if plan.save_distances
         append!(accept_log_combo.dist, accept_log.dist)
         append!(reject_log_combo.dist, reject_log.dist)
      end
      =#
  end
  weights = fill(1.0/plan.num_part,plan.num_part)
  logpriorvals = Distributions.logpdf(plan.prior,theta)

  #return abc_population_type(theta,weights,dist_theta,logpriorvals,summary_stat_log_combo)
  return abc_population_type(theta,weights,dist_theta,logpriorvals,accept_log_combo,reject_log_combo)
end

# Update the abc population once
function update_abc_pop_serial(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type, sampler::Distribution, epsilon::Float64;
                        attempts::Array{Int64,1} = zeros(Int64,plan.num_part) )
  #new_pop = deepcopy(pop)
  new_pop = abc_population_type(size(pop.theta,1), size(pop.theta,2), accept_log=pop.accept_log, reject_log=pop.reject_log, repeats=pop.repeats)
  if plan.save_params || plan.save_summary_stats || plan.save_distances
     push!(new_pop.accept_log.generation_starts_at,length(new_pop.accept_log.dist)+1)
     push!(new_pop.reject_log.generation_starts_at,length(new_pop.reject_log.dist)+1)
  end
     for i in 1:plan.num_part
       #theta_star, dist_theta_star, attempts[i], summary_stats = generate_theta(plan, sampler, ss_true, epsilon)
       theta_star, dist_theta_star, attempts[i], accept_log, reject_log = generate_theta(plan, sampler, ss_true, epsilon)
       append_to_abc_log!(new_pop.accept_log,plan,accept_log.theta,accept_log.ss,accept_log.dist,accept_log.full_model)
       append_to_abc_log!(new_pop.reject_log,plan,reject_log.theta,reject_log.ss,reject_log.dist,reject_log.full_model)
       #=
       if plan.save_params
          append!(new_pop.log.theta,summary_stats.theta)
       end
       if plan.save_summary_stats
          append!(new_pop.log.ss,summary_stats.ss)
       end
       if plan.save_distances
          #append!(new_pop.log.dist, dist_theta_star)
          append!(new_pop.log.dist, summary_stats.dist)
       end
       =#
       #if dist_theta_star < epsilon # replace theta with new set of parameters and update weight
         @inbounds new_pop.dist[i] = dist_theta_star
         @inbounds new_pop.theta[:,i] = theta_star
         prior_logpdf = Distributions.logpdf(plan.prior,theta_star)
         if isa(prior_logpdf, Array)   # TODO: Can remove this once Danley's code uses composite distribution
            prior_logpdf = prior_logpdf[1]
         end
         # sampler_pdf calculation must match distribution used to update particle
         sampler_logpdf = logpdf(sampler, theta_star )
         #@inbounds new_pop.weights[i] = prior_pdf/sampler_pdf
         @inbounds new_pop.weights[i] = exp(prior_logpdf-sampler_logpdf)
         @inbounds new_pop.logpdf[i] = sampler_logpdf
      if dist_theta_star < epsilon # replace theta with new set of parameters and update weight
         @inbounds new_pop.repeats[i] = 0
       else  # failed to generate a closer set of parameters, so...
         # ... generate new data set with existing parameters
         #new_data = plan.gen_data(theta_star)
         #new_ss = plan.calc_summary_stats(new_data)
         #@inbounds new_pop.dist[i] = plan.calc_dist(ss_true,new_ss)
         # ... just keep last value for this time, and mark it as a repeat
         #=
         @inbounds new_pop.dist[i] = pop.dist[i]
         @inbounds new_pop.theta[:,i] = pop.theta[:,i]
         @inbounds new_pop.weights[i] = pop.weights[i]
         @inbounds new_pop.logpdf[i] = pop.logpdf[i]
         =#
         @inbounds new_pop.repeats[i] += 1
       end
     end # i / num_parts
   new_pop.weights ./= sum(new_pop.weights)
   return new_pop
end

# run the ABC algorithm matching to summary statistics ss_true, starting from an initial population (e.g., output of previosu call)
function run_abc(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type; verbose::Bool = false, print_every::Integer=1, in_parallel::Bool = plan.in_parallel,
                 ss_scale::AbstractVector{Float64} = ones(length(ss_true)) )
  global num_emu_accepts, num_emu_rejects, num_emu_indecisive_span, num_emu_indecisive_cdf, num_emu_spans
  attempts = zeros(Int64,plan.num_part)
  # Set initial epsilon tolerance based on current population
  epsilon = quantile(pop.dist,plan.init_epsilon_quantile)
  local emu
   function emulate_ss(theta::Array{Float64,2})
      #global emu
      pred = ABC.predict_gp(emu,theta,sigmasq_obs=ss_scale.^2)
      #ABC.emulator_output_to_ss_mean(pred)
      ABC.emulator_output_to_ss_mean_stddev(pred)
  end
  function emulate_ss(theta::Array{Float64,1})
    #global emu
    pred = ABC.predict_gp(emu,reshape(theta,size(theta,1),1),sigmasq_obs=ss_scale.^2) #,calc_sigma_pred=calc_stddev_summary_stats_mean_stddev_rate_period_snr_detected)
    #ABC.emulator_output_to_ss_mean(pred)
    ABC.emulator_output_to_ss_mean_stddev(pred)
  end

  for t in 1:plan.num_max_times
    sampler = plan.make_proposal_dist(pop, plan.tau_factor)
    if plan.adaptive_quantiles
      min_quantile = t==1 ? 4.0/size(pop.theta,2) : 1.0/sqrt(size(pop.theta,2))
      epsilon = choose_epsilon_adaptive(pop, sampler, min_quantile=min_quantile)
    end
     if plan.use_emulator
        num_use = 200
        (x_train,y_train) = ABC.make_training_data(pop.accept_log, pop.reject_log, collect(1:length(pop.accept_log.generation_starts_at)), num_use)
        emu = ABC.optimize_gp_cor(x_train, y_train.mean,[ss_scale[i]^2 for i in 1:length(ss_scale), j in 1:size(x_train,2) ], param_init=[2.0,5.0])
        #emu = ABC.train_gp(x_train,y_train,sigmasq_cor=opt_out.minimum[1],rho=opt_out.minimum[2])

        #emu = ABC.train_gp(x_train,y_train)
        plan.emulator = emulate_ss
        num_emu_accepts = 0
        num_emu_rejects = 0
        num_emu_indecisive_span = 0
        num_emu_indecisive_cdf = 0
        num_emu_spans = zeros(Int64,6)
    end

    local new_pop
    if in_parallel
      #new_pop = update_abc_pop_parallel_darray(plan, ss_true, pop, epsilon, attempts=attempts)
	    new_pop = update_abc_pop_parallel_darray(plan, ss_true, pop, sampler, epsilon, attempts=attempts)
    else
	  #new_pop = update_abc_pop_serial(plan, ss_true, pop, epsilon, attempts=attempts)
      new_pop = update_abc_pop_serial(plan, ss_true, pop, sampler, epsilon, attempts=attempts)
    end
    pop = new_pop
    if verbose && (t%print_every == 0)
       println("# t= ",t, " eps= ",epsilon, " med(d)= ",median(pop.dist), " attempts= ",median(attempts), " ",maximum(attempts), " reps= ", sum(pop.repeats), " ess= ",ess(pop.weights,pop.repeats)) #," mean(theta)= ",mean(pop.theta,2) )#) #, " tau= ",diag(tau) ) #
       # println("# t= ",t, " eps= ",epsilon, " med(d)= ",median(pop.dist), " max(d)= ", maximum(pop.dist), " med(attempts)= ",median(attempts), " max(a)= ",maximum(attempts), " reps= ", sum(pop.repeats), " ess= ",ess(pop.weights,pop.repeats)) #," mean(theta)= ",mean(pop.theta,2) )#) #, " tau= ",diag(tau) ) #
       if plan.use_emulator
          println("# Emulator: ", num_emu_accepts, " ", num_emu_rejects, " ", num_emu_indecisive_span, " ", num_emu_indecisive_cdf, "  spans: ", num_emu_spans)
       end
    end
    #if epsilon < plan.target_epsilon  # stop once acheive goal
    if maximum(pop.dist) < plan.target_epsilon  # stop once acheive goal
       println("# Reached ",epsilon," after ", t, " generations.")
       break
    end
    if median(attempts)>0.1*plan.num_max_attempt
      println("# Halting due to ", median(attempts), " median number of valid attempts.")
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

function choose_epsilon_adaptive(pop::abc_population_type, sampler::Distribution; min_quantile::Real = 1.0/sqrt(size(pop.theta,2)) )
  sampler_logpdf = logpdf(sampler, pop.theta)
  target_quantile_this_itteration = min(1.0, exp(-maximum(sampler_logpdf .- pop.logpdf)) )
  if target_quantile_this_itteration > 1.0
     target_quantile_this_itteration = 1.0
  end
  optimal_target_quantile = target_quantile_this_itteration
  if target_quantile_this_itteration < min_quantile
     target_quantile_this_itteration = min_quantile
  end
  epsilon = quantile(pop.dist,target_quantile_this_itteration)
  println("# Estimated target quantile is ", optimal_target_quantile, " using ",target_quantile_this_itteration, " resulting in epsilon = ", epsilon)
  return epsilon
end

# run the ABC algorithm matching to summary statistics ss_true
function run_abc(plan::abc_pmc_plan_type, ss_true; verbose::Bool = false, print_every::Integer=1, in_parallel::Bool =  plan.in_parallel )                                          # Initialize population, drawing from prior
  pop::abc_population_type = init_abc(plan,ss_true, in_parallel=in_parallel)
  #println("pop_init: ",pop)
  run_abc(plan, ss_true, pop; verbose=verbose, print_every=print_every, in_parallel=in_parallel )  # Initialize population, drawing from prior
end

function emu_accept_prob(ss_true::Array{Float64,1},emu_out::mean_stddev_type{Array{Float64,1}}, epsilon::Real)
  @assert epsilon>0
  mu = emu_out.mean
  sig = emu_out.stddev
  prob_accept = 1.0
  for i in 1:length(ss_true)
    @assert sig[i]>0
    @assert dist_scale[i]>0
    distrib = Normal(mu[i],sig[i])
    cdf_hi = cdf(distrib,ss_true[i]+epsilon*dist_scale[i])
    cdf_lo = cdf(distrib,ss_true[i]-epsilon*dist_scale[i])
    prob_accept *= abs(cdf_hi-cdf_lo)
  end
  return prob_accept
end

num_emu_accepts = 0
num_emu_rejects = 0
num_emu_indecisive_span = 0
num_emu_indecisive_cdf = 0
num_emu_spans = zeros(Int64,6)
function emu_accept_reject_run_full_model(ss_true::Array{Float64,1},emu_out::mean_stddev_type{Array{Float64,1}}, epsilon::Real; prob_accept_crit::Real = 0.999, prob_reject_crit::Real = 0.001, sigma_factor::Real = 1.0)
  global num_emu_accepts, num_emu_rejects, num_emu_indecisive_span, num_emu_indecisive_cdf, num_emu_spans
  mu = emu_out.mean
  sig = emu_out.stddev
  run_full_model = true
  #param_list = []
  global num_param_active
  prob_accept = 1.0
  for i in 1:length(ss_true) # WARNING TODO RESTORE GENERAL
    distrib = Normal(mu[i],sig[i])
    cdf_hi = cdf(distrib,ss_true[i]+epsilon*dist_scale[i])
    cdf_lo = cdf(distrib,ss_true[i]-epsilon*dist_scale[i])
    prob_accept_this = abs(cdf_hi-cdf_lo)
    prob_accept *= prob_accept_this
    if ( (cdf_hi<prob_reject_crit) || (cdf_lo>prob_accept_crit) )
      run_full_model = false
      num_emu_rejects += 1
      num_emu_spans[i] +=1
      #println("# Rejecting emualated parameters w/ p=",prob_accept)
      #push!(param_list,i)
      return -1
    end
  end
  #prob_accept = emu_accept_prob(ss_true,emu_out,epsilon)
  if prob_accept > prob_accept_crit
    #println("# Accepting emualated parameters w/ p=",prob_accept)#, "emu_out = ",emu_out)
    num_emu_accepts += 1
    return 1
  elseif  prob_accept <= prob_reject_crit
    num_emu_indecisive_cdf += 1
    #println("# Rejecting emualated parameters w/ p=",prob_accept)
    return 0
  else
    #println("# Need to rerun full model due to   p=",prob_accept)
    num_emu_indecisive_cdf += 1
    return 0
  end
end


# run the ABC algorithm matching to summary statistics ss_true
function run_abc_emulated(plan::abc_pmc_plan_type, ss_true; verbose::Bool = false, print_every::Integer=1, num_use::Integer = 500, in_parallel::Bool =  plan.in_parallel )                                          # Initialize population, drawing from prior
  pop_out = init_abc(plan,ss_true, in_parallel=in_parallel)
  #println("pop_init: ",pop)
  (x_train,y_train) = make_training_data(pop_out.accept_log, pop_out.reject_log, collect(1:length(pop_out.accept_log.generation_starts_at)), num_use)
  emu = ABC.train_gp(x_train,y_train)
  function emulate_ss(theta::Array{Float64,2})
   pred = ABC.predict_gp(emu,theta)
   ABC.emulator_output_to_ss_mean(pred)
  end
  function emulate_ss(theta::Array{Float64,1})
   pred = ABC.predict_gp(emu,reshape(theta,size(theta,1),1))
   ABC.emulator_output_to_ss_mean(pred)
  end

  emu_out = emulate_ss(theta)
  plan.calc_summary_stats = emulate_ss

  run_abc(plan, ss_true, pop; verbose=verbose, print_every=print_every, in_parallel=in_parallel )  # Initialize population, drawing from prior
end

#=
pop_out = init_abc(abc_plan,ss_true)
num_use = 500
  (x_train,y_train) = ABC.make_training_data(pop_out.accept_log, pop_out.reject_log, [length(pop_out.accept_log.generation_starts_at)], num_use)
  emu = ABC.train_gp(x_train,y_train)
  function emulate_ss(theta::Array{Float64,2})
   pred = ABC.predict_gp(emu,theta)
   #ABC.emulator_output_to_ss_mean(pred)
   ABC.emulator_output_to_ss_mean_stddev(pred)
  end
  function emulate_ss(theta::Array{Float64,1})
   pred = ABC.predict_gp(emu,reshape(theta,size(theta,1),1))
   #ABC.emulator_output_to_ss_mean(pred)
   ABC.emulator_output_to_ss_mean_stddev(pred)
  end

  emu_out = emulate_ss(theta_true)
  abc_plan.emulator = emulate_ss
  abc_plan.use_emulator = true
  abc_plan.num_max_times = 1
  pop_emu = run_abc(abc_plan, ss_true, pop_out; verbose=true, print_every=1)


=#
