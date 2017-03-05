using Distributions
using PDMats
#import Compat.view # For backward compatability with v0.4
include(joinpath(Pkg.dir("ABC"),"src/composite.jl"))  # Not yet put in it's own package
using CompositeDistributions

# Set HyperPrior for Population Parameters
param_prior_rate = Gamma(1.0,1.0)
 param_prior_mean  = MvLogNormal(log([10.0, 10.0]), diagm(log([10.0,10.0])))
 param_prior_covar_diag = MvLogNormal(zeros(2),ones(2))
 #param_prior_cor = MvNormal(zeros(1),ones(1));
 param_prior = CompositeDist( ContinuousDistribution[param_prior_rate,param_prior_mean,param_prior_covar_diag] )
 theta_true = [1.0, log(10.0), log(10.0), log(10.0), log(10.0)] # True Population Parameters
 num_param_active = 5
 num_star_default = 1000   # How many "stars" to include in universe

 function is_valid_mean_covar(theta::Array{Float64,1})  # Ensure that the proposed covariance matrix is positive definite
  if length(theta) >= num_param_active+1    theta[num_param_active+1:end] = theta_true[num_param_active+1:end]   end
  a = theta[4]^2
  d = theta[5]^2
  #b = theta[4]*theta[5]*theta[6]
  #c = theta[4]*theta[5]*theta[6]
  tr_covar = a+d
  det_covar = a*d # -b*c
  if (det_covar>0) && (tr_covar>0)
    return true
  else
    return false
  end
 end

 function is_valid_model(theta::Array{Float64,1})
    if length(theta) >= num_param_active+1   theta[num_param_active+1:end] = theta_true[num_param_active+1:end]   end
    if any(theta[1:5] .< 0.0) return false end
    if !is_valid_mean_covar(theta) return false end
    return true
 end


function detection_prob_snr_term(snr::Real) # Detection probability as a function of signal-to-noise ratio
  const sig =  1.0
  const snr_thresh = 7.1
  const snr_detect_never =  3.0
  const snr_detect_plateau = 10.0
  const sqrt2 = sqrt(2.0)
  const pdet_max = 0.5*( 1+erf((snr_detect_plateau - snr_thresh)/(sqrt2*sig)) )
  pdet = 0.0
  if snr > snr_detect_plateau
     pdet = pdet_max
  elseif snr > snr_detect_never
     pdet = 0.5*( 1+erf((snr - snr_thresh)/(sqrt2*sig)) )
  end
  return pdet
 end

 function detection_prob_geometry_term(period::Real) # Transit probability as a function of orbital geometry
    const radius_star = 0.005
    const days_in_year = 365.2425
    a = (period/days_in_year)^(2//3)
    prob_geometric = min(1.0,radius_star/a)
 end

 detection_prob_return_one(period::Real, snr::Real) = 1.0
 detection_prob_snr_only(period::Real, snr::Real) = detection_prob_snr_term(snr)
 detection_prob_geometry_only(period::Real, snr::Real) = detection_prob_geometry_term(period)
 detection_prob_snr_and_geometry(period::Real, snr::Real) = detection_prob_geometry_term(period) * detection_prob_snr_term(snr)

 # Code to generate simulated data given array of model parameters
 function gen_number_planets(theta::Array, num_stars::Integer)
    rate = num_param_active>=1 ? theta[1]^2 : theta_true^2
    if rate<=0.0  return 0 end
    sum(rand(Poisson(rate),num_stars))
 end
 function gen_period_snr(theta::Array, n::Integer)  # Just return "planet" properties
  if length(theta) >= num_param_active+1   theta[num_param_active+1:end] = theta_true[num_param_active+1:end]   end
  mean = theta[2:3]
  rho = 0. # theta[6]
  covar = PDMat([theta[4]^2 theta[4]*theta[5]*rho; theta[4]*theta[5]*rho theta[5]^2])
  period_snr = rand(MvLogNormal(mean,covar),n)
  return period_snr
 end

 function gen_period_snr_detected(theta::Array, num_stars::Integer; detection_prob::Function = detection_prob_snr_and_geometry)  # "Planets" detection efficient depends on SNR and orbital period
  n = gen_number_planets(theta,num_stars)
  period_snr = gen_period_snr(theta,n)
  detected = trues(n)
  for i in 1:n
    period = period_snr[1,i]
    snr = period_snr[2,i]
    prob_detect = detection_prob( period, snr )
    if rand() > prob_detect
      detected[i] = false
    end
  end
  return (period_snr, detected)
 end

 gen_period_snr_detected_all(theta::Array, n::Integer = num_star_default) = gen_period_snr_detected(theta,n,detection_prob=detection_prob_return_one)
 gen_period_snr_detected_snr_only(theta::Array, n::Integer = num_star_default) = gen_period_snr_detected(theta,n,detection_prob=detection_prob_snr_only)
 gen_period_snr_detected_geometry_only(theta::Array, n::Integer = num_star_default) = gen_period_snr_detected(theta,n,detection_prob=detection_prob_geometry_only)
 gen_period_snr_detected_snr_and_geo(theta::Array, n::Integer = num_star_default) = gen_period_snr_detected(theta,n,detection_prob=detection_prob_snr_and_geometry)


function calc_mean_summary_stats_mean_stddev_rate_period_snr_detected(data::Tuple{Array{Float64,2},BitArray{1}})
  period_snr = data[1]
  detected = data[2]
  @assert size(period_snr,1) == 2
  @assert size(period_snr,2) == length(detected)
  num_detected = sum(detected)
  lambda = num_detected/num_star_default
  if num_detected<1
    return [0.0,Inf,Inf,Inf,Inf,Inf][1:num_param_active]
  end
  logPdet = log(period_snr[1,detected])
  logSnrdet = log(period_snr[2,detected])
  mean_log_period = mean(logPdet)
  mean_log_snr = mean(logSnrdet)
  if num_detected<2
     return [sqrt(lambda),mean_log_period,mean_log_snr,Inf,Inf] #
  end
  stddev_log_period = stdm(logPdet,mean_log_period)
  stddev_log_snr = stdm(logSnrdet,mean_log_snr)
  #covar_cross = sum( (logPdet.-mean_log_period) .* (logSnrdet.-mean_log_snr) )/(num_detected-1)
  #cor =  covar_cross / ( stddev_log_period*stddev_log_snr*(num_detected-1) )
  #cor_F = -1<cor<1 ? atanh(cor) : (cor<0?-Inf:Inf)
  return [sqrt(lambda), mean_log_period, mean_log_snr, stddev_log_period, stddev_log_snr]
end

function calc_stddev_summary_stats_mean_stddev_rate_period_snr_detected(ss::Array{Float64,1} )
  sqrt_planets_per_star = ss[1]
  planets_per_star = sqrt_planets_per_star^2
  n = num_star_default*planets_per_star
  one_over_sqrtn = 1.0/sqrt(n)
  if num_param_active == 1
    return [0.5*one_over_sqrtn]
  end
  sd_logp = ss[4]
  sd_logsnr = ss[5]
  const two_to_quarter = 2^(1//4)
  return [0.5*one_over_sqrtn, sd_logp*one_over_sqrtn, sd_logsnr*one_over_sqrtn, sd_logp*two_to_quarter*one_over_sqrtn, sd_logsnr*two_to_quarter*one_over_sqrtn]
end

function calc_stddev_summary_stats_mean_stddev_rate_period_snr_detected(data::Tuple{Array{Float64,2},BitArray{1}}, ss::Array{Float64,1} )
  calc_stddev_summary_stats_mean_stddev_rate_period_snr_detected(ss)
end

function calc_mean_stddev_summary_stats_mean_stddev_rate_period_snr_detected(data::Tuple{Array{Float64,2},BitArray{1}})
  mean_ss = calc_mean_summary_stats_mean_stddev_rate_period_snr_detected(data)
  stddev_ss = calc_stddev_summary_stats_mean_stddev_rate_period_snr_detected(data, mean_ss)
  return ABC.mean_stddev_type(mean_ss,stddev_ss)
end

proposal(pop::abc_population_type, tau_factor::Float64) = ABC.make_proposal_dist_gaussian_subset_full_covar(pop, tau_factor, param_active = collect(1:num_param_active) )

# Uncomment one of three options below one to pick how data is generated
#gen_data = gen_period_snr_detected_all
#num_star_default = 2000   # How many "planets" to include in universe
#gen_data = gen_period_snr_detected_snr_only
#num_star_default = 4000   # How many "planets" to include in universe
gen_data = gen_period_snr_detected_snr_and_geo
num_star_default = 160000   # How many "planets" to include in universe

#=
ex_data =gen_data(theta_true)
calc_summary_stats_mean_stddev_detected(ex_data)
length(find(ex_data[2]))
=#

# Tell ABC what it needs to know for a simulation
abc_plan = ABC.abc_pmc_plan_type(gen_data,
                                 #calc_mean_summary_stats_mean_stddev_rate_period_snr_detected,
                                 calc_mean_stddev_summary_stats_mean_stddev_rate_period_snr_detected,
                                 #ABC.calc_dist_max, param_prior;target_epsilon=0.001,epsilon_init=10.0,num_part=40,
                                 ABC.calc_scaled_dist_max, param_prior;target_epsilon=12.0,epsilon_init=300.0,num_part=60,
                             #is_valid=is_valid_mean_covar,adaptive_quantiles=true,num_max_attempt=1000,
                                 is_valid=is_valid_model,adaptive_quantiles=true,num_max_attempt=4000,
    make_proposal_dist =proposal);
   #make_proposal_dist = ABC.make_proposal_dist_gaussian_full_covar);
#= abc_plan = abc_pmc_plan_type(gen_data,calc_summary_stats_mean_stddev_detected,ABC.calc_scaled_dist_max, param_prior;
                             is_valid=is_valid_mean_covar,adaptive_quantiles=true,num_max_attempt=1000,
                             target_epsilon=0.01,epsilon_init=10.0,num_part=50,
   make_proposal_dist = ABC.make_proposal_dist_gaussian_full_covar); =#

