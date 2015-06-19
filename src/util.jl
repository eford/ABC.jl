# If want to use an empty function for normalizing/testing validity of parameters
noop(x::Array) = return true

# Copy population
import Base.copy
copy(x::abc_population_type) = abc_population_type(copy(x.theta),copy(x.weights),copy(x.dist), copy(x.repeats))

# Compute weighted covariance of sample (from origin) 
function cov_weighted(x::Array{Float64,2}, w::Array{Float64,1} )
  @assert(size(x,1)==length(w) )
  sumw = sum(w)
  if(sumw!= 1.0) 
     w /= sum(w)
     sumw = 1.0
  end
  sumw2 = sum(w.*w)  
  xbar = [ sum(x[:,i].*w) for i in 1:size(x,2) ]  
  covar = zeros(size(x,2),size(x,2))
  for k in 1:size(x,2)
    for j in 1:size(x,2)
        for i in 1:size(x,1)
            @inbounds covar[j,k] += (x[i,j]-xbar[j])*(x[i,k]-xbar[k]) *w[i]
        end
    @inbounds covar[j,k] *= sumw/(sumw*sumw-sumw2)
    end
  end
  covar
end

# Compute Effective Sample Size for array of weights
function ess(w::Array{Float64,1})
  sumw = sum(w)
  sumw2 = sum (w.*w)
  return sumw*sumw/sumw2
end


# Compute Effective Sample Size for array of weights, ignorring any repeated elements
function ess(w::Array{Float64,1}, repeat::Array{Int64,1} )
  @assert(length(w)==length(repeat))
  sumw = sum(w[find(x->x==0,repeat)])
  sumw2 = sum (w[find(x->x==0,repeat)].^2)
  return sumw*sumw/sumw2
end


# Common summary stats and distances
calc_summary_stats_mean_var(x::Array) = ( @inbounds m=mean(x); @inbounds v = varm(x,m); return [m, v] )

calc_dist_max(x::Array{Float64,1},y::Array{Float64,1}) = maximum(abs(x.-y))

# Summary Stats & Distances based on Empirical CDF
# using Distributions
calc_summary_stats_ecdf(x::Array) = EmpiricalUnivariateDistribution(x)


function calc_dist_ks(x::EmpiricalUnivariateDistribution, y::EmpiricalUnivariateDistribution)
   maxd = 0.0
   for v in x.values
     d = abs(cdf(x,v)-cdf(y,v))
     if d>maxd maxd = d end
   end   
   for v in y.values
     d = abs(cdf(x,v)-cdf(y,v))
     if d>maxd maxd = d end
   end   
   maxd
end


