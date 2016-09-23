abstract abc_plan_type

type abc_pmc_plan_type <: abc_plan_type
   gen_data::Function
   calc_summary_stats::Function
   calc_dist::Function                         
   prior::Distribution
   make_proposal_dist::Function
   normalize::Function
   is_valid::Function
   num_part::Integer
   num_max_attempt::Integer 
   num_max_times::Integer 
   epsilon_init::Float64 
   init_epsilon_quantile::Float64 
   epsilon_reduction_factor::Float64 
   target_epsilon::Float64 
   tau_factor::Float64 
   param_active::Vector{Int64}   # Not implemented yet
   save_summary_stats::Bool   # Not implemented yet
   in_parallel::Bool 
   
   function abc_pmc_plan_type(gd::Function,css::Function,cd::Function,p::Distribution;                          
     make_proposal_dist::Function = make_proposal_dist_gaussian_full_covar, param_active::Vector{Int64} = collect(1:length(Distributions.rand(p))), 
     normalize::Function = noop, is_valid::Function = noop, 
     num_part::Integer = 10*length(Distributions.rand(p))^2, num_max_attempt::Integer = 1000, num_max_times::Integer = 100,
     epsilon_init::Float64 = 1.0, init_epsilon_quantile::Float64 = 0.75, epsilon_reduction_factor::Float64 = 0.9, 
     target_epsilon::Float64 = 0.01, tau_factor::Float64 = 2.0, 
     save_summary_stats::Bool = true, in_parallel::Bool = false)
     @assert(num_part>=length(Distributions.rand(p)))
     @assert(num_max_attempt>=1)
     @assert(num_max_times>0)
     @assert(epsilon_init>0.01)
     @assert(0<init_epsilon_quantile<=1.0)
     @assert(0.5<epsilon_reduction_factor<1.0)
     @assert(target_epsilon>0.0)
     @assert(1.0<=tau_factor<=4.0)
     new(gd,css,cd,p,make_proposal_dist,normalize,is_valid, num_part,num_max_attempt,num_max_times,epsilon_init, init_epsilon_quantile,epsilon_reduction_factor,target_epsilon,tau_factor,param_active,save_summary_stats,in_parallel)
   end
end

type abc_summary_stats_log_type   # Not implemented yet
   theta::Array{Array{Float64,1},1}
   ss::Array{Array{Float64},1}

   function abc_summary_stats_log_type(t::Array{Array{Float64,1},1}, ss::Array{Array{Float64,1},1})
      @assert length(t) == length(ss)
      new( t, ss )
   end
end

function abc_summary_stats_log_type()
   abc_summary_stats_log_type( Array(Array{Float64,1},0), Array(Array{Float64,1},0) )
end


type abc_population_type
   theta::Array{Float64,2}
   weights::Array{Float64,1}
   dist::Array{Float64,1}
   log::abc_summary_stats_log_type
   repeats::Array{Int64,1}

   function abc_population_type(t::Array{Float64,2}, w::Array{Float64,1}, d::Array{Float64,1}, l::abc_summary_stats_log_type = abc_summary_stats_log_type(), repeats::Array{Int64,1} = zeros(Int64,length(w)) ) 
      @assert(length(w)==length(d)==size(t,2)==length(repeats))
      new(t,w,d,l,repeats)
   end                                    
end


