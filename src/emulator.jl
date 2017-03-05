if !isdefined(:MultivariateStats) using MultivariateStats end
if !isdefined(:PDMats) using PDMats end
if !isdefined(:Distances) using Distances end
if !isdefined(:Optim) using Optim end
import StatsBase.mean

type emulator_gp_type{T<:Real}
  sigmasq_cor::T
  rho::T
  kernel::Function
  input_train::Array{T,2}
  input_means::Array{T,1}
  input_covar::PDMat
  output_train::Array{T,2}
  output_means::Array{T,1}
  Sigma_train::Array{PDMat,1}
end

function emulator_gp_type()
  T = Float64
  emulator_gp_type(one(T),one(T),kernel_exp,Array(T,(0,0)),Array(T,0),PDMat(ones(0,0)),Array(T,(0,0)),Array(T,0),Array(PDMat,0))
end


type emulator_gp_savable_type{T<:Real}
  sigmasq_cor::T
  rho::T
  kernel::String                   # JLD can't save a function
  input_train::Array{T,2}
  input_means::Array{T,1}
  input_covar::PDMat
  output_train::Array{T,2}
  output_means::Array{T,1}
  Sigma_train::Array{Array{T,2},1} # JLD doesn't like saving an Array of PDMats
end
function emulator_gp_savable_type{T<:Real}(emu::emulator_gp_type{T})
  emulator_gp_savable_type(emu.sigmasq_cor,emu.rho,string(emu.kernel),emu.input_train,emu.input_means,emu.input_covar,emu.output_train,emu.output_means,Matrix{T}[emu.Sigma_train[i].mat for i in 1:length(emu.Sigma_train)])
end

function emulator_gp_type{T<:Real}(emu::emulator_gp_savable_type{T})
  emulator_gp_type(emu.sigmasq_cor,emu.rho,eval(Symbol(emu.kernel)),emu.input_train,emu.input_means,emu.input_covar,emu.output_train,emu.output_means,PDMat[PDMat(emu.Sigma_train[i]) for i in 1:length(emu.Sigma_train)])
end

function kernel_exp(d::Float64; rho::Float64 = 1.0)
   exp(-d/rho)
end

function kernel_exp(d::Array{Float64}; rho::Float64 = 1.0)
   exp(-d/rho)
end

function make_kernel_data{T<:Real}(x::AbstractArray{T,2}, kernel::Function;
        #param_means::AbstractArray{T,1} = vec(mean(x,2)), param_covar::AbstractArray{T,2} = Base.covm(x',param_means'),
        param_means::AbstractArray{T,1} = vec(mean(x,2)), param_covar::PDMat = PDMat(Base.covm(x',param_means')),
			sigmasq_obs::Array{T,1} = zeros(T,size(x,2)), sigmasq_cor::T = one(T), rho::T = one(T))
  param_dist_sq = pairwise(SqMahalanobis(inv(param_covar).mat),x.-param_means)
  #K = param_dist_sq
  K  = diagm(sigmasq_obs)
  K += sigmasq_cor*kernel(param_dist_sq,rho=rho^2)
  return K
end

function make_kernel_data{T<:Real}(x::AbstractArray{T,2}, xstar::AbstractArray{T,2}, kernel::Function;
        #param_means::AbstractArray{T,1} = vec(mean(x,2)), param_covar::AbstractArray{T,2} = Base.covm(x',param_means'),
        param_means::AbstractArray{T,1} = vec(mean(x,2)), param_covar::PDMat = PDMat(Base.covm(x',param_means')),
			  sigmasq_cor::T = one(T), rho::T = one(T))
  param_dist_sq = pairwise(SqMahalanobis(inv(param_covar).mat),x.-param_means,xstar.-param_means)
  K = sigmasq_cor*kernel(param_dist_sq,rho=rho^2)
  return K
end

function softabs(hessian; alpha::Float64 = 1000.0)
  lam, Q = eig(hessian)
  lam_twig = lam./tanh(alpha*lam)
  #H_twig = PDMat(Q*diagm(lam_twig)*Q')
  H_twig = PDMat(X_A_Xt(PDiagMat(lam_twig),Q))
end

function train_gp{T<:Real}(x::AbstractArray{T,2}, y::mean_stddev_type{Array{T,2}}; kernel::Function = kernel_exp,
               sigmasq_cor::T = one(T), rho::T = one(T), sigma_x_nugget::Float64 = 1e-6 )
  xm = vec(mean(x,2))
  xc = PDMat(Base.covm(x',xm')+diagm(sigma_x_nugget*ones(size(x,1))))
  num_outputs = size(y.mean,1)
  Sigma_tr = Array(PDMat,num_outputs)
  for i in 1:num_outputs   # TODO WARNING Update how sigmasq_obs is calculated
    mat = make_kernel_data(x,kernel,param_means=xm,param_covar=xc,sigmasq_obs=y.stddev[i,:].^2,sigmasq_cor=sigmasq_cor,rho=rho)
    Sigma_tr[i] = softabs(mat)
    #Sigma_tr[i] = PDMat(mat)
  end
  #  ym = vec(mean(y.mean,weights(y.stddev.^(-2)),2))
  ym = vec(mean(y.mean,2))

  return emulator_gp_type(sigmasq_cor,rho,kernel,x,xm,xc,y.mean,ym,Sigma_tr)
end

function train_gp{T<:Real}(x::AbstractArray{T,2}, y::AbstractArray{T,2}, sigmasq_y::AbstractArray{T,2}; kernel::Function = kernel_exp,
               sigmasq_cor::T = one(T), rho::T = one(T), sigma_x_nugget::Float64 = 1e-6  )
  xm = vec(mean(x,2))
  xc = PDMat(Base.covm(x',xm')+diagm(sigma_x_nugget*ones(size(x,1))))
  num_outputs = size(y,1)
  Sigma_tr = Array(PDMat,num_outputs)
  for i in 1:num_outputs
    mat = make_kernel_data(x,kernel,param_means=xm,param_covar=xc,sigmasq_obs=sigmasq_y[i,:],sigmasq_cor=sigmasq_cor,rho=rho)
    Sigma_tr[i] = softabs(mat)
  end
  ym = vec(mean(y,2))
  return emulator_gp_type(sigmasq_cor,rho,kernel,x,xm,xc,y,ym,Sigma_tr)
end

function marginal_gp(emulator::emulator_gp_type, index::Integer)
  -0.5*( invquad(emulator.Sigma_train[index], emulator.output_train[index,:].-emulator.output_means[index]) + logdet(emulator.Sigma_train[index]) + size(emulator.input_train,2)*log(2pi) )
end

function grad_marginal_gp(emulator::emulator_gp_type, index::Integer)
  @assert 1<=index<=length(emulator.Sigma_train)
  # WARNING: Need to generalize, currently assumes ExpSquared Kernel
  alpha = emulator.Sigma_train[index] \ (emulator.output_train[index,:].-emulator.output_means[index])
  # 0.5*trace(alpha*alpha'*dKdtheta - Kinv \ dKdtheta)
  if index==1
    rho = emulator.rho
    # dKdtheta  = emulator.Sigma_train[index].mat/rho
    dlogpdtheta = trace(alpha*alpha'*emulator.Sigma_train[index].mat - diagm(ones(length(alpha))))*0.5/rho
  elseif index==2
    sigma = emulator.sigmasq_cor
    d = sqrt(pairwise(SqMahalanobis(inv(emulator.input_covar).mat),emulator.input_train.-emulator.input_means))
    #dKdtheta  = -2*d/rho^2*emulator.Sigma_train[index].mat
    #param_dist_sq = pairwise(SqMahalanobis(inv(emulator.Sigma_train[index]).mat),emulator.input_train.-emulator.input_means)
    #dKdtheta  = -2*sqrt(param_dist_sq)/rho^2*emulator.Sigma_train[index].mat
    dlogpdtheta = -trace(alpha*alpha'*(d*emulator.Sigma_train[index].mat) - d*diagm(ones(length(alpha)))) /sigma^2
  end
  return dlogpdtheta
end

function marginal_gp(emulator::emulator_gp_type)
  logprob = 0.
  for i in 1:size(emulator.output_train,1)
    logprob += marginal_gp(emulator,i)
  end
  return logprob
end

function grad_marginal_gp(emulator::emulator_gp_type)
  gradlogprob = 0.
  for i in 1:size(emulator.output_train,1)
    gradlogprob += grad_marginal_gp(emulator,i)
  end
  return gradlogprob
end

function optimize_gp_cor_brute_force{T<:Real}(x::AbstractArray{T,2}, y::AbstractArray{T,2}, sigmasq_y::AbstractArray{T,2}; kernel::Function = kernel_exp )
  function calc_marginal(param::Array{T,1})
    emu = train_gp(x,y,sigmasq_y,kernel=kernel,sigmasq_cor=param[1],rho=param[2])
    marginal_gp(emu)
  end
  corlist = [0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.]
  rholist = [0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.]
  cor_best = 0.0
  rho_best = 0.0
  marginal_best = -Inf
  for i in 1:length(corlist)
  for j in 1:length(rholist)
      marg = calc_marginal([corlist[i],rholist[j]])
      println("# sigma_c=",corlist[i], " rho=",rholist[j]," logp=",marg)
      if marg > marginal_best
        cor_best = corlist[i]
        rho_best = rholist[j]
        marginal_best = marg
      end
    end
 end
 println("# Max GP marginal at: ",cor_best," ", rho_best)
 train_gp(x,y,sigmasq_y,kernel=kernel,sigmasq_cor=cor_best,rho=rho_best)
end


function optimize_gp_cor{T<:Real}(x::AbstractArray{T,2}, y::AbstractArray{T,2}, sigmasq_y::AbstractArray{T,2}; kernel::Function = kernel_exp, param_init::AbstractArray{T,1} = ones(2), LBFGS::Bool = false )
  param_last = copy(param_init)    # zeros(T,size(x,1))
  emulator = train_gp(x,y,sigmasq_y,kernel=kernel,sigmasq_cor=abs(param_init[1]),rho=abs(param_init[2]))
  function train_if_changed!(param::Array{T,1}, param_last::Array{T,1}, emulator::emulator_gp_type)
    if param != param_last
      copy!(param_last,param)
      emulator = train_gp(x,y,sigmasq_y,kernel=kernel,sigmasq_cor=abs(param[1]),rho=abs(param[2]))
    end
  end
  function calc_marginal(param::Array{T,1})
    train_if_changed!(param,param_last,emulator)
    -marginal_gp(emulator)+sum((param.^2))
  end
  function calc_grad_marginal!(param::Array{T,1},storage::Vector)
    train_if_changed!(param,param_last,emulator)
    storage[1] = -sign(param[1])*grad_marginal_gp(emulator,1)+2*param[1]
    storage[2] = -sign(param[2])*grad_marginal_gp(emulator,2)+2*param[2]
    return storage
  end

  if LBFGS
    opt_out = Optim.optimize(calc_marginal,calc_grad_marginal!,param_init,Optim.LBFGS(),Optim.OptimizationOptions(show_trace=true,show_every=1,extended_trace=true)) #, x_tol=0.1))
  else
    opt_out = Optim.optimize(calc_marginal,calc_grad_marginal!,param_init,Optim.BFGS(),Optim.OptimizationOptions(show_trace=true,show_every=1,extended_trace=true)) #, x_tol=0.1))
  end
  return emulator
end


function predict_gp{T<:Real}(emulator::emulator_gp_type, x_star::AbstractArray{T,2};
         sigmasq_obs::Array{T,1} = zeros(T,(size(emulator.output_train,1))) )
  sigmasq_obs_x_star = [sigmasq_obs[i] for i in 1:size(emulator.output_train,1), j in 1:size(x_star,2)]
  predict_gp(emulator,x_star,sigmasq_obs_x_star)
end

function predict_gp_mean{T<:Real}(emulator::emulator_gp_type, x_star::AbstractArray{T,2};
         sigmasq_obs::Array{T,1} = zeros(T,(size(emulator.output_train,1))) )
  sigmasq_obs_x_star = [sigmasq_obs[i] for i in 1:size(emulator.output_train,1), j in 1:size(x_star,2)]
  predict_gp_mean(emulator,x_star,sigmasq_obs_x_star)
end

function predict_gp{T<:Real}(emulator::emulator_gp_type, x_star::AbstractArray{T,2}, sigmasq_obs::Array{T,2} )
  Sigma_c = make_kernel_data(emulator.input_train,x_star,emulator.kernel,param_means=emulator.input_means,param_covar=emulator.input_covar,sigmasq_cor=emulator.sigmasq_cor,rho=emulator.rho)
  Sigma_pred = make_kernel_data(x_star,emulator.kernel,param_means=emulator.input_means,param_covar=emulator.input_covar,sigmasq_cor=emulator.sigmasq_cor,rho=emulator.rho)
  num_outputs = size(emulator.output_train,1)
  predict_mean = Array(T,(num_outputs,size(x_star,2)))
  predict_covar = Array(T,(num_outputs,size(x_star,2),size(x_star,2)))
  for i in 1:num_outputs
    predict_mean[i,:] = Sigma_c' * (emulator.Sigma_train[i] \ (emulator.output_train[i,:].-emulator.output_means[i])) .+ emulator.output_means[i]
    # TODO WARNING Update how sigmasq_obs is calculated to include covariances
    predict_covar[i,:,:] = Sigma_pred - Sigma_c' * (emulator.Sigma_train[i] \ Sigma_c)+ diagm(sigmasq_obs[i,:])
  end
  #=
  for j in 1:size(predict_mean,2)
     sigma_obs = calc_sigma_pred(predict_mean[:,j])
     predict_covar[:,j,j] += sigmasq_obs[i,:]
  end
  =#
  return (predict_mean, predict_covar)
end

function predict_gp_mean{T<:Real}(emulator::emulator_gp_type, x_star::AbstractArray{T,2}, sigmasq_obs::Array{T,2} )
  Sigma_c = make_kernel_data(emulator.input_train,x_star,emulator.kernel,param_means=emulator.input_means,param_covar=emulator.input_covar,sigmasq_cor=emulator.sigmasq_cor,rho=emulator.rho)
  Sigma_pred = make_kernel_data(x_star,emulator.kernel,param_means=emulator.input_means,param_covar=emulator.input_covar,sigmasq_cor=emulator.sigmasq_cor,rho=emulator.rho)
  num_outputs = size(emulator.output_train,1)
  predict_mean = Array(T,(num_outputs,size(x_star,2)))
  predict_covar = Array(T,(num_outputs,size(x_star,2),size(x_star,2)))
  for i in 1:num_outputs
    predict_mean[i,:] = Sigma_c' * (emulator.Sigma_train[i] \ (emulator.output_train[i,:].-emulator.output_means[i])) .+ emulator.output_means[i]
    # TODO WARNING Update how sigmasq_obs is calculated to include covariances
    predict_covar[i,:,:] = Sigma_pred - Sigma_c' * (emulator.Sigma_train[i] \ Sigma_c)+ diagm(sigmasq_obs[i,:])
  end
  #=
  for j in 1:size(predict_mean,2)
     sigma_obs = calc_sigma_pred(predict_mean[:,j])
     predict_covar[:,j,j] += sigmasq_obs[i,:]
  end
  =#
  return (predict_mean, predict_covar)
end

 function emulator_output_to_ss_mean( emu_output::Tuple{Array{Float64,2},Array{Float64,3}} )
   vec(emu_output[1])
 end

 function emulator_output_to_ss_mean_stddev( emu_output::Tuple{Array{Float64,2},Array{Float64,3}} )
   m = emu_output[1]
   c = emu_output[2]
   stddev = similar(m)
   for i in 1:size(m,1)
     stddev[i,:] = sqrt(diag(c[i,:,:]))
     #stddev[i,:] = sqrt(diag(c[i,:,:]) + stddev_prediction^2)  # TODO:  How to get this here in a generic way?
   end
   return mean_stddev_type(vec(m),vec(stddev))
 end


function test_emulator()
  num_train = 100
  num_test = 101
  num_inputs = 3
  num_outputs = 2
  x = rand(num_inputs,num_train)
  xs = rand(num_inputs,num_test)
  A = zeros(num_outputs,num_inputs)
  for i in 1:min(num_outputs,num_inputs)
    A[i,i] = 1.0
  end
  A += 0.1randn(num_outputs,num_inputs)
  b = 3.0*rand(num_outputs)
  sigma = 0.1
  y  = b.+A*x .+sigma*rand(num_outputs,num_train)
  ys = b.+A*xs.+sigma*rand(num_outputs,num_test)
  emu = ABC.train_gp(x,y,sigma^2*reshape(ones(length(y)),size(y)))

  mar = ABC.marginal_gp(emu)
  (em_y, em_sigma) = ABC.predict_gp(emu,xs)

  resid = em_y-ys
  resid_std = similar(resid)
  for i in 1:size(ys,1)
    resid_std[i,:] = (em_y[i,:]-ys[i,:])./diag(em_sigma[i,:,:])
  end
  resid_std

  index_x = 1
  index_y = 1
  plot(vec(x[index_x,:]),vec(y[index_y,:]),"c.")
  plot(vec(xs[index_x,:]),vec(ys[index_y,:]),"b.")
  plt[:errorbar](vec(xs[index_x,:]),vec(em_y[index_y,:]),yerr=diag(em_sigma[index_y,:,:]),fmt=".")

  # plot residuals
  for index_x in 1:size(xs,1)
     for index_y in 1:size(ys,1)
       #plot(vec(xs[index_x,:]),vec(em_y[index_y,:]).-vec(ys[index_y,:]),".")
       plt[:errorbar](vec(xs[index_x,:]),vec(em_y[index_y,:]).-vec(ys[index_y,:]),yerr=diag(em_sigma[index_y,:,:]),fmt=".")
    end
  end

end

#=
(x_train,y_train) = ABC.make_training_data(pop_out.accept_log,pop_out.reject_log,10,100)
emu = ABC.optimize_gp_cor(x_train,y_train,y_train.+1.0)
x_pred = x_train
(y_pred,sigma_y) = ABC.predict_gp(emu,x_pred)
plot(x_train[1,:],y_train[3,:],"b.")
plot(x_pred[2,:],y_pred[3,:],"r.")

=#

function test_abc_emulator(accept_log::ABC.abc_log_type, reject_log::ABC.abc_log_type, generation::Integer, num_use::Integer = 10)
  (x_train,y_train) = make_training_data(accept_log, reject_log, generation, num_use)
  emu = train_gp(x_train,y_train,y_train)
  mar = marginal_gp(emu)
  println("# logprob = ",mar)
  x_pred = x_train
  y_pred = predict_gp(emu,x_pred)
  return (x_train,y_train,x_pred,y_pred)
end
