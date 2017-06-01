
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
  #summary_stat_logs = Array(abc_log_type,plan.num_part)
  #summary_stat_log_combo = abc_log_type()
  accept_log_combo = abc_log_type()
  reject_log_combo = abc_log_type()
  if plan.save_params || plan.save_summary_stats || plan.save_distances
     #push!(summary_stat_log_combo.generation_starts_at,length(summary_stat_log_combo.dist)+1)
     push!(accept_log_combo.generation_starts_at,length(accept_log_combo.dist)+1)
     push!(reject_log_combo.generation_starts_at,length(reject_log_combo.dist)+1)
  end
  for i in 1:plan.num_part
      theta[:,i]  = map_results[i][1]
      dist_theta[i]  = map_results[i][2]
      attempts[i]  = map_results[i][3]
      if plan.save_params || plan.save_summary_stats || plan.save_distances
         #summary_stat_logs[i] = map_results[i][4]
         accept_log = map_results[i][4]
         reject_log = map_results[i][5]
         append_to_abc_log!(accept_log_combo,plan,accept_log.theta,accept_log.ss,accept_log.dist)
         append_to_abc_log!(reject_log_combo,plan,reject_log.theta,reject_log.ss,reject_log.dist)
         #=
         if plan.save_params
           append!(summary_stat_log_combo.theta, summary_stat_logs[i].theta)
        end
        if plan.save_summary_stats
          append!(summary_stat_log_combo.ss, summary_stat_logs[i].ss)
        end
        if plan.save_distances
           append!(summary_stat_log_combo.dist, summary_stat_logs[i].dist)
        end
        =#
      end
  end

  weights = fill(1.0/plan.num_part,plan.num_part)
  logpriorvals = Distributions.logpdf(plan.prior,theta)
  #return abc_population_type(theta,weights,dist_theta,logpriorvals,summary_stat_log_combo)
  return abc_population_type(theta,weights,dist_theta,plan.epsilon_init,logpriorvals,accept_log_combo,reject_log_combo)
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
  #summary_stat_logs = Array(abc_log_type,plan.num_part)
  #summary_stat_log_combo = abc_log_type()
  accept_log_combo = abc_log_type()
  reject_log_combo = abc_log_type()
  if plan.save_params || plan.save_summary_stats || plan.save_distances
     push!(summary_stat_log_combo.generation_starts_at,length(summary_stat_log_combo.dist)+1)
  end
  for i in 1:plan.num_part
      theta[:,i]  = pmap_results[i][1]
      dist_theta[i]  = pmap_results[i][2]
      attempts[i]  = pmap_results[i][3]
      #=if plan.save_summary_stats
         summary_stat_logs[i] = pmap_results[i][4]
         append!(summary_stat_log_combo.theta, summary_stat_logs[i].theta)
         append!(summary_stat_log_combo.ss, summary_stat_logs[i].ss)
      end
      =#
         accept_log = pmap_results[i][4]
         reject_log = pmap_results[i][5]
         append_to_abc_log!(accept_log_combo,plan,accept_log.theta,accept_log.ss,accept_log.dist)
         append_to_abc_log!(reject_log_combo,plan,reject_log.theta,reject_log.ss,reject_log.dist)
      #=
      if plan.save_params || plan.save_summary_stats || plan.save_distances
         summary_stat_logs[i] = pmap_results[i][4]
         if plan.save_params
           append!(summary_stat_log_combo.theta, summary_stat_logs[i].theta)
        end
        if plan.save_summary_stats
          append!(summary_stat_log_combo.ss, summary_stat_logs[i].ss)
        end
        if plan.save_distances
           append!(summary_stat_log_combo.dist, summary_stat_logs[i].dist)
        end
      end
      =#
  end

  weights = fill(1.0/plan.num_part,plan.num_part)
  logpriorvals = Distributions.logpdf(plan.prior,theta)
  #return abc_population_type(theta,weights,dist_theta,plan.epsilon_init,logpriorvals,summary_stat_log_combo)
  return abc_population_type(theta,weights,dist_theta,plan.epsilon_init,logpriorvals,accept_log_combo,reject_log_combo)
end

# Update the abc population once
function update_abc_pop_parallel_pmap(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type, sampler::Distribution, epsilon::Float64;
                        attempts::Array{Int64,1} = zeros(Int64,plan.num_part))
  #new_pop = deepcopy(pop)
  new_pop = abc_population_type(size(pop.theta,1), size(pop.theta,2), accept_log=pop.accept_log, reject_log=pop.reject_log, repeats=pop.repeats)
  pmap_results = pmap(x->generate_theta(plan, sampler, ss_true, epsilon), collect(1:plan.num_part))
  if plan.save_params || plan.save_summary_stats || plan.save_distances
     push!(new_pop.accept_log.generation_starts_at,length(new_pop.accept_log.dist)+1)
     push!(new_pop.reject_log.generation_starts_at,length(new_pop.reject_log.dist)+1)
  end
     for i in 1:plan.num_part
       #theta_star, dist_theta_star, attempts[i] = generate_theta(plan, sampler, ss_true, epsilon)
       # if dist_theta_star < pop.dist[i] # replace theta with new set of parameters and update weight
       theta_star = pmap_results[i][1]
       dist_theta_star =  pmap_results[i][2]
       accept_log = pmap_results[i][4]
       reject_log = pmap_results[i][5]
       append_to_abc_log!(new_pop.accept_log,plan,accept_log.theta,accept_log.ss,accept_log.dist)
       append_to_abc_log!(new_pop.reject_log,plan,reject_log.theta,reject_log.ss,reject_log.dist)

       #=
       if plan.params
          append!(new_pop.log.theta, pmap_results[i][4].theta)
       end
       if plan.save_summary_stats
          append!(new_pop.log.ss, pmap_results[i][4].ss)
       end
       if plan.save_distances
          append!(new_pop.log.dist, pmap_results[i][4].dist)
       end
       =#
       if dist_theta_star < epsilon # replace theta with new set of parameters and update weight
         @inbounds new_pop.theta[:,i] = theta_star
         @inbounds new_pop.dist[i] = dist_theta_star
         prior_logpdf = Distributions.logpdf(plan.prior,theta_star)
         if isa(prior_pdf, Array)
            prior_logpdf = prior_logpdf[1]
         end
         # sampler_pdf calculation must match distribution used to update particle
         sampler_logpdf = pdf(sampler, theta_star )
         @inbounds new_pop.weights[i] = exp(prior_logpdf-sampler_logpdf)
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

function update_abc_pop_parallel_darray(plan::abc_pmc_plan_type, ss_true, pop::abc_population_type, sampler::Distribution, epsilon::Float64;
                        attempts::Array{Int64,1} = zeros(Int64,plan.num_part))
  #new_pop = deepcopy(pop)
  new_pop = abc_population_type(size(pop.theta,1), size(pop.theta,2), accept_log=pop.accept_log, reject_log=pop.reject_log, repeats=pop.repeats)
  darr_silly = dzeros(plan.num_part)
  dmap_results = map(x->generate_theta(plan, sampler, ss_true, epsilon), darr_silly)
  map_results = collect(dmap_results)
  if plan.save_params || plan.save_summary_stats || plan.save_distances
     push!(new_pop.accept_log.generation_starts_at,length(new_pop.accept_log.dist)+1)
     push!(new_pop.reject_log.generation_starts_at,length(new_pop.reject_log.dist)+1)
  end
  for i in 1:plan.num_part
       #theta_star, dist_theta_star, attempts[i] = generate_theta(plan, sampler, ss_true, epsilon)
       # if dist_theta_star < pop.dist[i] # replace theta with new set of parameters and update weight
       theta_star = map_results[i][1]
       dist_theta_star =  map_results[i][2]
       accept_log = map_results[i][4]
       reject_log = map_results[i][5]
       append_to_abc_log!(new_pop.accept_log,plan,accept_log.theta,accept_log.ss,accept_log.dist)
       append_to_abc_log!(new_pop.reject_log,plan,reject_log.theta,reject_log.ss,reject_log.dist)

       #=
       if plan.save_params
          append!(new_pop.log.theta,dmap_results[i][4].theta)
       end
       if plan.save_summary_stats
          append!(new_pop.log.ss,dmap_results[i][4].ss)
       end
       if plan.save_distances
          append!(new_pop.log.dist, dmap_results[i][4].dist)
       end
       =#
       if dist_theta_star < epsilon # replace theta with new set of parameters and update weight
         @inbounds new_pop.theta[:,i] = theta_star
         @inbounds new_pop.dist[i] = dist_theta_star
         prior_logpdf = Distributions.pdf(plan.prior,theta_star)
         if isa(prior_logpdf, Array)
            prior_logpdf = prior_logpdf[1]
         end
         # sampler_pdf calculation must match distribution used to update particle
         sampler_logpdf = logpdf(sampler, theta_star )
         @inbounds new_pop.weights[i] = exp(prior_logpdf-sampler_logpdf)
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




