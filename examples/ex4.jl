workspace()
using ABC
using Distributions
using PDMats
include(joinpath(Pkg.dir("ABC"),"src/composite.jl"))  # Not yet put in it's own package
using CompositeDistributions
import Compat.view # For backward compatability with v0.4

# Set HyperPrior for Population Parameters
param_prior_mean  = MvLogNormal(log([10.0, 10.0]), diagm(log([10.0,10.0])))
 param_prior_covar_diag = MvLogNormal(zeros(2),ones(2))
 param_prior_cor = MvNormal(zeros(1),ones(1));
 param_prior = CompositeDist( ContinuousDistribution[param_prior_mean,param_prior_covar_diag,param_prior_cor] )

function detection_prob(snr::Real) # Detection probability as a function of signal-to-noise ratio
  const sig =  1.0
  const snr_thresh = 7.1
  const snr_detect_never =  3.0
  const snr_detect_always = 10.0
  const sqrt2 = sqrt(2.0)
  pdet = 0.0
  if snr > snr_detect_always
     pdet = 1.0
  elseif snr > snr_detect_never
     pdet = 0.5*( 1+erf((snr - snr_thresh)/(sqrt2*sig)) )
  end
  return pdet
end

num_data_default = 1000   # How many "planets" to include in universe
 # Code to generate simulated data given array of model parameters
 function gen_period_snr(theta::Array, n::Integer = num_data_default)  # Just return "planet" properties
  mean = theta[1:2]
  covar = PDMat([theta[3]^2 theta[3]*theta[4]*theta[5]; theta[3]*theta[4]*theta[5] theta[4]^2])
  period_snr = rand(MvLogNormal(mean,covar),n)
  return period_snr
 end

 function gen_period_snr_detected_all(theta::Array, n::Integer = num_data_default)  # All "planets" are detected
  return (gen_period_snr(theta,n), trues(n))
 end

 function gen_period_snr_detected_snr_only(theta::Array, n::Integer = num_data_default) # "Planets" detection efficient depends on SNR only
  period_snr = gen_period_snr(theta,n)
  detected = trues(n)
  for i in 1:n
    snr = period_snr[2,i]
    prob_detect = detection_prob( snr )
    if rand() > prob_detect
      detected[i] = false
    end
  end
  return (period_snr, detected)
 end

 function gen_period_snr_detected_snr_and_geo(theta::Array, n::Integer = num_data_default)  # "Planets" detection efficient depends on SNR and orbital period
  period_snr = gen_period_snr(theta,n)
  detected = trues(n)
  const radius_star = 0.005
  const days_in_year = 365.2425
  for i in 1:n
    period = period_snr[1,i]
    snr = period_snr[2,i]
    prob_detect = detection_prob( snr )
    a = (period/days_in_year)^(2//3)
    prob_geometric = min(1.0,radius_star/a)
    prob_detect *= prob_geometric
    if rand() > prob_detect
      detected[i] = false
    end
  end
  return (period_snr, detected)
 end


 function is_valid_mean_covar(theta::Array{Float64,1})  # Ensure that the propose covariance matrix is positive definite
  a = theta[3]^2                     # WARNING: Covariance matrix extracted here and when generating data must match
  b = theta[3]*theta[4]*theta[5]
  c = theta[3]*theta[4]*theta[5]
  d = theta[4]^2
  tr_covar = a+d
  det_covar = a*d-b*c
  if (det_covar>0) && (tr_covar>0)
    return true
  else
    return false
  end
end

function calc_summary_stats_mean_stddev_detected(data::Tuple{Array{Float64,2},BitArray{1}})
  period_snr = data[1]
  detected = data[2]
  @assert size(period_snr,1) == 2
  @assert size(period_snr,2) == length(detected)
  if length(find(detected))<2
     return [Inf,Inf,Inf,Inf,Inf]
  end
  mean_log_period = mean(log(period_snr[1,detected]))
  stddev_log_period = stdm(log(period_snr[1,detected]),mean_log_period)
  mean_log_snr = mean(log(period_snr[2,detected]))
  stddev_log_snr = stdm(log(period_snr[2,detected]),mean_log_snr)
  covar_cross = sum( (log(period_snr[1,detected]).-mean_log_period) .* (log(period_snr[2,detected]).-mean_log_snr) )/length(find(detected))
  cor =  covar_cross / (stddev_log_period*stddev_log_snr)
  return [mean_log_period, stddev_log_period, mean_log_snr, stddev_log_snr, cor]

end

theta_true = [log(10.0), log(10.0), log(10.0), log(10.0), 0.0] # True Population Parameters
#=
theta_true = fill(NaN,5) #  Generate from hyperprior
 while !is_valid_mean_covar(theta_true)
  theta_true = rand(param_prior)
 end
 theta_true
=#

# Uncomment one of three options below one to pick how data is generated
#gen_data = gen_period_snr_detected_all
#num_data_default = 4000   # How many "planets" to include in universe
#gen_data = gen_period_snr_detected_snr_only
#num_data_default = 8000   # How many "planets" to include in universe
gen_data = gen_period_snr_detected_snr_and_geo
num_data_default = 48000   # How many "planets" to include in universe

#=
ex_data =gen_data(theta_true)
calc_summary_stats_mean_stddev_detected(ex_data)
length(find(ex_data[2]))
=#

# Tell ABC what it needs to know for a simulation
abc_plan = abc_pmc_plan_type(gen_data,calc_summary_stats_mean_stddev_detected,ABC.calc_dist_max, param_prior;
                             is_valid=is_valid_mean_covar,adaptive_quantiles=true,num_max_attempt=1000,
                             target_epsilon=0.01,epsilon_init=10.0,num_part=100,
   make_proposal_dist = ABC.make_proposal_dist_gaussian_full_covar);

# Generate "true/observed data"
data_true = gen_data(theta_true)   # Draw "real" data from same model as for analysis
ss_true = abc_plan.calc_summary_stats(data_true)

#=
pdf(abc_plan.prior,theta_true)
CompositeDistributions.index(abc_plan.prior,1)
CompositeDistributions.index(abc_plan.prior,2)
logpdf(abc_plan.prior.dist[1],theta_true[CompositeDistributions.index(abc_plan.prior,1)])
logpdf(abc_plan.prior.dist[2],theta_true[CompositeDistributions.index(abc_plan.prior,2)])

logpdf(abc_plan.prior.dist[1],theta_true[1:2])
logpdf(abc_plan.prior.dist[2],theta_true[3:5])
println("***************************************************")
=#
# Run ABC simulation
@time pop_out = run_abc(abc_plan,ss_true, verbose=true);

using JLD
save("ex4_out.jld", "pop_out", pop_out)

using PyPlot
function plot_abc_posterior(pop::ABC.abc_population_type, index::Integer )
  lo = minimum(pop_out.theta[index,:])
  hi = maximum(pop_out.theta[index,:])
  #sigma = (hi-lo)/length(pop.weights)
  sigma = 3.0*sqrt(abs(cov(vec(pop_out.theta[index,:]),pop.weights)))
  x = collect(linspace(lo-3*sigma,hi+3*sigma,100))
  y = zeros(x)
  for i in 1:length(x)
    for j in 1:size(pop.theta,2)
      y[i] += pop.weights[j]*pdf(Normal(pop.theta[index,j],sigma),x[i])
    end
  end
  plot(x,y,"-")
end
plot_abc_posterior(pop_out,1)
plot_abc_posterior(pop_out,2)
plot_abc_posterior(pop_out,3)
plot_abc_posterior(pop_out,4)
plot_abc_posterior(pop_out,5)

