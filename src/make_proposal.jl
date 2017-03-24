function make_proposal_dist_gaussian_full_covar(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false, param_active = nothing, max_maha_distsq_per_dim::Real = 0.0)
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
  if max_maha_distsq_per_dim > 0
    max_maha_distsq = max_maha_distsq_per_dim*size(theta_mean,1)
    sampler = GaussianMixtureModelCommonCovarTruncated(pop.theta,pop.weights,covar,max_maha_distsq)
  else
    sampler = GaussianMixtureModelCommonCovar(pop.theta,pop.weights,covar)
  end
  return sampler
end

function make_proposal_dist_gaussian_diag_covar(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false, param_active = nothing, max_maha_distsq_per_dim::Real = 0.0 )
  theta_mean = sum(pop.theta.*pop.weights') # weighted mean for parameters
  tau = tau_factor*var_weighted(pop.theta'.-theta_mean,pop.weights)  # scaled, weighted covar for parameters
  if verbose
    println("theta_mean = ", theta_mean)
    println("pop.theta = ", pop.theta)
    println("pop.weights = ", pop.weights)
    println("tau = ", tau)
  end
  covar = PDiagMat(tau)
  if max_maha_distsq_per_dim > 0
    max_maha_distsq = max_maha_distsq_per_dim*size(theta_mean,1)
    sampler = GaussianMixtureModelCommonCovarTruncated(pop.theta,pop.weights,covar,max_maha_distsq)
  else
    sampler = GaussianMixtureModelCommonCovar(pop.theta,pop.weights,covar)
  end
end

function make_proposal_dist_gaussian_subset_full_covar(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false, param_active::Vector{Int64} = collect(1:size(pop.covar,1)) )
  theta_mean = sum(pop.theta.*pop.weights') # weighted mean for parameters
  rawtau = cov_weighted(pop.theta'.-theta_mean,pop.weights)  # scaled, weighted covar for parameters
  tau = tau_factor*make_matrix_pd(rawtau)
  if verbose
    println("theta_mean = ", theta_mean)
    println("pop.theta = ", pop.theta)
    println("pop.weights = ", pop.weights)
    println("tau = ", tau)
  end
  covar = tau # PDMat(tau[param_active])
  sampler = GaussianMixtureModelCommonCovarSubset(pop.theta,pop.weights,covar,param_active)
end

function make_proposal_dist_gaussian_subset_diag_covar(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false, param_active::Vector{Int64} = collect(1:size(pop.covar,1)) )
  theta_mean = sum(pop.theta.*pop.weights') # weighted mean for parameters
  tau = tau_factor*var_weighted(pop.theta'.-theta_mean,pop.weights)  # scaled, weighted covar for parameters
  if verbose
    println("theta_mean = ", theta_mean)
    println("pop.theta = ", pop.theta)
    println("pop.weights = ", pop.weights)
    println("tau = ", tau)
  end
  covar = tau
  sampler = GaussianMixtureModelCommonCovarSubset(pop.theta,pop.weights,covar,param_active)
end

function make_proposal_dist_gaussian_rand_subset_full_covar(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false, num_param_active::Integer = 2)
  param_active = union(sample(1:size(pop.theta,1),num_param_active,replace=false))
  make_proposal_dist_gaussian_subset_full_covar(pop,tau_factor, verbose=verbose, param_active=param_active )
end

function make_proposal_dist_gaussian_rand_subset_diag_covar(pop::abc_population_type, tau_factor::Float64; verbose::Bool = false, num_param_active::Integer = 2)
  param_active = union(sample(1:size(pop.theta,1),num_param_active,replace=false))
  make_proposal_dist_gaussian_subset_diag_covar(pop,tau_factor, verbose=verbose, param_active=param_active )
end

