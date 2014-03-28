abstract abc_plan_type

type abc_pmc_plan_type <: abc_plan_type
   gen_data::Function
   calc_summary_stats::Function
   calc_dist::Function                         
   prior::Distribution
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
   
   function abc_pmc_plan_type(gd::Function,css::Function,cd::Function,p::Distribution;                          
     normalize::Function = noop, is_valid::Function = noop, 
     num_part::Integer = 10*length(Distributions.rand(p))^2, num_max_attempt::Integer = 1000, num_max_times::Integer = 100,
     epsilon_init::Float64 = 1.0, init_epsilon_quantile::Float64 = 0.75, epsilon_reduction_factor::Float64 = 0.9, 
     target_epsilon::Float64 = 0.01, tau_factor::Float64 = 2.0)
     @assert(num_part>=length(Distributions.rand(p)))
     @assert(num_max_attempt>=1)
     @assert(num_max_times>0)
     @assert(epsilon_init>0.01)
     @assert(0<init_epsilon_quantile<=1.0)
     @assert(0.5<epsilon_reduction_factor<1.0)
     @assert(target_epsilon>0.0)
     @assert(1.0<=tau_factor<=4.0)
     new(gd,css,cd,p,normalize,is_valid,num_part,num_max_attempt,num_max_times,epsilon_init, init_epsilon_quantile,epsilon_reduction_factor,target_epsilon,tau_factor)
   end
end


type abc_population_type
   theta::Array{Float64,2}
   weights::Array{Float64,1}
   dist::Array{Float64,1}
   repeats::Array{Int64,1}
   function abc_population_type(t::Array{Float64,2}, w::Array{Float64,1}, d::Array{Float64,1}, r::Array{Int64,1} = zeros(Int64,length(w)) )
      @assert(length(w)==length(d)==size(t,2)==length(r))
      new(t,w,d,r)                                  
   end                                    
end

