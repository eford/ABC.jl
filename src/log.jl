
function push_to_abc_log!(abc_log::abc_log_type, plan::abc_pmc_plan_type, theta_star::Array{Float64,1}, ss_star, dist_star::Float64)
         if plan.save_params
           push!(abc_log.theta, theta_star)
         end
         if plan.save_summary_stats
           push!(abc_log.ss, ss_star)
         end
         if plan.save_distances
           push!(abc_log.dist, dist_star)
         end
end

function append_to_abc_log!(abc_log::abc_log_type, plan::abc_pmc_plan_type, theta_star::Array{Array{Float64,1},1}, ss_star::Array{Any,1}, dist_star::Array{Float64,1})
         if plan.save_params
           append!(abc_log.theta, theta_star)
         end
         if plan.save_summary_stats
           append!(abc_log.ss, ss_star)
         end
         if plan.save_distances
           append!(abc_log.dist, dist_star)
         end
end

include("emulator.jl")

function draw_indices(abc_log::ABC.abc_log_type, num_use::Integer, start_idx::Integer = 1, stop_idx::Integer = length(abc_log.dist))
  @assert num_use >=1
  num_available = 1+stop_idx-start_idx
  idx = collect(start_idx:min(start_idx+num_available-1,start_idx+num_use-1))
end

function make_training_data(accept_log::ABC.abc_log_type, reject_log::ABC.abc_log_type, generation::Integer, num_use::Integer = 10)
 @assert length(accept_log.theta) == length(accept_log.ss) == length(accept_log.dist) >= 1
 @assert length(reject_log.theta) == length(reject_log.ss) == length(reject_log.dist) >= 1
 @assert length(accept_log.theta[1]) >=1
 @assert length(reject_log.theta[1]) >=1
 @assert length(accept_log.ss[1]) == length(reject_log.ss[1])
 @assert length(accept_log.generation_starts_at) == length(reject_log.generation_starts_at) >= generation >=1
 num_use_accepts = num_use
 num_use_rejects = num_use
 stop_idx_accept = generation+1 <= length(accept_log.generation_starts_at) ? accept_log.generation_starts_at[generation+1] : length(accept_log.dist)
 stop_idx_reject = generation+1 <= length(reject_log.generation_starts_at) ? reject_log.generation_starts_at[generation+1] : length(reject_log.dist)
 idx_accept = draw_indices(accept_log,num_use_accepts,accept_log.generation_starts_at[generation],stop_idx_accept)
 idx_reject = draw_indices(reject_log,num_use_rejects,reject_log.generation_starts_at[generation],stop_idx_reject)
 num_use_total = length(idx_accept) + length(idx_reject)
 num_param = length(accept_log.theta[1])
 num_ss = length(accept_log.ss[1])
 x_train = Array(Float64,(num_param,num_use_total))
 y_train = Array(Float64,(num_ss,num_use_total))
 println("# idx_accept = ",idx_accept)
 println("# size(x_train)",size(x_train))
 println("# size(accept_log.theta[idx_accept]) = ", size( accept_log.theta[idx_accept]))
 for i in 1:length(idx_accept)
    x_train[:,i] = accept_log.theta[idx_accept[i]]
    y_train[:,i] = accept_log.ss[idx_accept[i]]
 end
 for i in 1:length(idx_reject)
    x_train[:,length(idx_accept)+i] = reject_log.theta[idx_reject[i]]
    y_train[:,length(idx_accept)+i] = reject_log.ss[idx_reject[i]]
 end
 return (x_train, y_train)
end

function test_abc_emulator(accept_log::ABC.abc_log_type, reject_log::ABC.abc_log_type, generation::Integer, num_use::Integer = 10)
  (x_train,y_train) = make_training_data(accept_log, reject_log, generation, num_use)
  emu = train_gp(x_train,y_train,y_train)
  mar = marginal_gp(emu)
  println("# logprob = ",mar)
  x_pred = x_train
  y_pred = predict_gp(emu,x_pred)
  return (x_train,y_train,x_pred,y_pred)
end
