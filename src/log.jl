
function push_to_abc_log!(abc_log::abc_log_type, plan::abc_pmc_plan_type, theta_star::Array{Float64,1}, ss_star, dist_star::Float64, full::Bool)
         if plan.save_params
           push!(abc_log.theta, theta_star)
         end
         if plan.save_summary_stats
           push!(abc_log.ss, ss_star)
         end
         if plan.save_distances
           push!(abc_log.dist, dist_star)
         end
         if plan.save_params || plan.save_summary_stats || plan.save_distances || plan.save_distances
           push!(abc_log.full_model, full)
         end
end

function append_to_abc_log!(abc_log::abc_log_type, plan::abc_pmc_plan_type, theta_star::Array{Array{Float64,1},1}, ss_star::Array{Any,1}, dist_star::Array{Float64,1}, full::Array{Bool,1})
         if plan.save_params
           append!(abc_log.theta, theta_star)
         end
         if plan.save_summary_stats
           append!(abc_log.ss, ss_star)
         end
         if plan.save_distances
           append!(abc_log.dist, dist_star)
         end
         if plan.save_params || plan.save_summary_stats || plan.save_distances || plan.save_distances
           append!(abc_log.full_model, full)
         end
end


function draw_indices(abc_log::ABC.abc_log_type, num_try_to_use::Integer; start::Integer = 1, stop::Integer = length(abc_log.dist))
  @assert num_try_to_use >=1
  #num_available = 1+min(stop_idx,length(abc_log.dist)-start_idx)
  #@assert num_available >=1
  #idx = collect(start_idx:min(start_idx+num_available-1,start_idx+num_use-1))
  #idx_range = start_idx:min(start_idx+num_available-1,start_idx+num_use-1)
  idx_range = start:stop
  all_good = trues(length(idx_range))
  for i in 1:length(all_good)
    idx = idx_range[i]
    if isnan(abc_log.dist[idx])                all_good[i] = false end
    if !all(isfinite(abc_log.ss[idx].mean))    all_good[i] = false end
    if !all(isfinite(abc_log.ss[idx].stddev))  all_good[i] = false end
    if !abc_log.full_model[idx]                all_good[i] = false end
  end
  idx_valid = find(all_good)
  num_sample = min(num_try_to_use,length(idx_valid))
  @assert num_sample >=1
  idx_sample = sort(sample(idx_valid,num_sample,replace=false))
  #idx_sorted_dist = sortperm(abc_log.dist[idx_valid])
  #idx_sample = idx_valid[idx_sorted_dist[1:num_sample]]
  return idx_range[idx_sample]
end

function draw_indicies_from_generations(accept_log::ABC.abc_log_type, reject_log::ABC.abc_log_type, generation_list::Vector{Int64}, num_use::Integer = 100)
 @assert length(generation_list) >= 1
 num_use_accepts = num_use
 num_use_rejects = num_use
 idx_accept = Int64[]
 idx_reject = Int64[]
 sorted_gen_list = sort(generation_list,rev=true)
 for generation in sorted_gen_list
    if (generation<1)||(generation>length(accept_log.generation_starts_at)) continue end
    @assert length(accept_log.generation_starts_at) == length(reject_log.generation_starts_at) >= generation >=1
    stop_idx_accept = generation+1 <= length(accept_log.generation_starts_at) ? accept_log.generation_starts_at[generation+1]-1 : length(accept_log.dist)
    idx_accept_this_gen = draw_indices(accept_log,num_use_accepts,start=accept_log.generation_starts_at[generation],stop=stop_idx_accept)
    append!(idx_accept,idx_accept_this_gen)
    num_use_accepts -= length(idx_accept_this_gen)
    if num_use_accepts <= 0 break end
 end
 num_use_rejects = num_use_rejects #+ num_use_accepts - length(idx_accept)
 for generation in sorted_gen_list
    if (generation<1)||(generation>length(reject_log.generation_starts_at)) continue end
    @assert length(accept_log.generation_starts_at) == length(reject_log.generation_starts_at) >= generation >=1
    stop_idx_reject = generation+1 <= length(reject_log.generation_starts_at) ? reject_log.generation_starts_at[generation+1]-1 : length(reject_log.dist)
    idx_reject_this_gen = draw_indices(reject_log,num_use_rejects,start=reject_log.generation_starts_at[generation],stop=stop_idx_reject)
    append!(idx_reject,idx_reject_this_gen)
    num_use_rejects -= length(idx_reject_this_gen)
    if num_use_rejects <= 0 break end
 end
 return (idx_accept,idx_reject)
end

function make_training_data(accept_log::ABC.abc_log_type, reject_log::ABC.abc_log_type, generation_list::Vector{Int64}, num_use::Integer = 10)
 @assert length(accept_log.theta) == length(accept_log.ss) == length(accept_log.dist) >= 1
 @assert length(reject_log.theta) == length(reject_log.ss) == length(reject_log.dist) >= 1
 @assert length(accept_log.theta[1]) >=1
 @assert length(reject_log.theta[1]) >=1
 if isa(accept_log.ss[1], ABC.mean_stddev_type)
   @assert length(accept_log.ss[1].mean) == length(reject_log.ss[1].mean)
   num_ss = length(accept_log.ss[1].mean)
 else
   @assert length(accept_log.ss[1]) == length(reject_log.ss[1])
   num_ss = length(accept_log.ss[1])
 end
 (idx_accept,idx_reject) = ABC.draw_indicies_from_generations(accept_log,reject_log,generation_list,num_use)
 println("# Training emulator on ", length(idx_accept), " accepts and ", length(idx_reject)," rejects.")
 num_use_total = length(idx_accept) + length(idx_reject)
 num_param = length(accept_log.theta[1])

 x_train = Array(Float64,(num_param,num_use_total))
 y_train = Array(Float64,(num_ss,num_use_total))
 if isa(accept_log.ss[1], ABC.mean_stddev_type)
    stddev_y_train = Array(Float64,(num_ss,num_use_total))
 end
 #println("# idx_accept = ",idx_accept)
 #println("# size(x_train)",size(x_train))
 #println("# size(accept_log.theta[idx_accept]) = ", size( accept_log.theta[idx_accept]))
 for i in 1:length(idx_accept)
    x_train[:,i] = accept_log.theta[idx_accept[i]]
    if isa(accept_log.ss[1], ABC.mean_stddev_type)
      y_train[:,i] = accept_log.ss[idx_accept[i]].mean
      stddev_y_train[:,i] = accept_log.ss[idx_accept[i]].stddev
    else
      y_train[:,i] = accept_log.ss[idx_accept[i]]
    end
 end
 for i in 1:length(idx_reject)
    x_train[:,length(idx_accept)+i] = reject_log.theta[idx_reject[i]]
    if isa(accept_log.ss[1], ABC.mean_stddev_type)
      y_train[:,length(idx_accept)+i] = reject_log.ss[idx_reject[i]].mean
      stddev_y_train[:,length(idx_accept)+i] = reject_log.ss[idx_reject[i]].stddev
    else
      y_train[:,length(idx_accept)+i] = reject_log.ss[idx_reject[i]]
    end
 end
  if isa(accept_log.ss[1], ABC.mean_stddev_type)
   return (x_train, ABC.mean_stddev_type(y_train,stddev_y_train))
  else
    return (x_train, y_train)
  end
end

###=
function test_emu(pop_out::abc_population_type, theta::Array{Float64,1}, num_use::Integer = 100)
 num_use = 100
 (x_train,y_train) = ABC.make_training_data(pop_out.accept_log, pop_out.reject_log, collect(1:length(pop_out.accept_log.generation_starts_at)), num_use)
 #=
 (x_train,y_train) = ABC.make_training_data(pop_out.accept_log, pop_out.reject_log, [length(pop_out.accept_log.generation_starts_at)], num_use)
 sample_var_y_train = var(y_train,2)
 sigmasq_y_train = similar(y_train)
 for i in 1:size(y_train,2)
   sigmasq_y_train[:,i] = sample_var_y_train
 end
 emu = ABC.train_gp(x_train,y_train,sigmasq_y_train)
  =#
 emu = ABC.train_gp(x_train,y_train)
 function emulate_ss(theta::Array{Float64,2})
   pred = ABC.predict_gp(emu,theta)
   emulator_output_to_ss_mean_stddev(pred)
 end
 function emulate_ss(theta::Array{Float64,1})
   pred = ABC.predict_gp(emu,reshape(theta,size(theta,1),1))
   emulator_output_to_ss_mean_stddev(pred)
 end

  emu_out = emulate_ss(theta)
end


#=
x = collect(linspace(0.0,0.1,200))
y=  map(x->emu_accept_reject(ss_true,emu_out,x),x)
plot(x,y)
=#
