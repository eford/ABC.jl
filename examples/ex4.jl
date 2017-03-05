workspace()
using ABC

include("../examples/ex5_model5.jl")
# Generate "true/observed data"
#theta_true = draw_theta_valid(abc_plan,abc_plan.prior)
data_true = gen_data(theta_true)   # Draw "real" data from same model as for analysis
#ss_true = abc_plan.calc_summary_stats(data_true)
ss_true = abc_plan.calc_summary_stats(data_true).mean
ss_scale = abc_plan.calc_summary_stats(data_true).stddev + 1e-4
ABC.set_distance_scale_max(abc_plan, theta_true, ss =ss_true )

# Run ABC simulation
target_epsilon_full = 1.0
target_epsilon_emu = 1.0
abc_plan.num_part = 50

println("# Now computing initial population")
@time pop_init = init_abc(abc_plan,ss_true);

if true
println("# Now training emulator")
num_use = 100
  #(x_train,y_train) = ABC.make_training_data(pop_out.accept_log, pop_out.reject_log, collect(1:length(pop_out.accept_log.generation_starts_at)), num_use)
  (x_train,y_train) = ABC.make_training_data(pop_init.accept_log, pop_init.reject_log, collect(1:length(pop_init.accept_log.generation_starts_at)), num_use)
  #emu = ABC.train_gp(x_train,y_train)
  #emu = ABC.optimize_gp_cor(x_train, y_train.mean,[ss_scale[i]^2 for i in 1:length(ss_scale), j in 1:size(x_train,2) ], param_init=[2.0,5.0])
  emu = ABC.train_gp(x_train,y_train,sigmasq_cor=2.0,rho=5.0)
end

println("# Now running with emulator")
abc_plan_emu = deepcopy(abc_plan)
abc_plan_emu.use_emulator = true
abc_plan_emu.num_max_times = 20
abc_plan_emu.target_epsilon = target_epsilon_emu
@time pop_emu = run_abc(abc_plan_emu, ss_true, pop_init; verbose=true, print_every=1, ss_scale=ss_scale);

#quit()

println("# Now running without emulator")
abc_plan.target_epsilon = target_epsilon_full
abc_plan_emu.use_emulator = false
abc_plan.num_max_times = 20
@time pop_out = run_abc(abc_plan, ss_true, pop_init; verbose=true, print_every=1, ss_scale=ss_scale);

num_plot = 200
if true
println("# Now resampling with emulator")
@time (theta_emu, d_emu) = ABC.generate_abc_sample(abc_plan, pop_emu, ss_true, target_epsilon_emu, num_plot=num_plot);
theta_mean = sum(theta_emu,2)/size(theta_emu,2)
println("theta_mean = ",theta_mean)
theta_stddev = sqrt(ABC.var_weighted(theta_emu'.-theta_mean',ones(size(theta_emu,2))))  # scaled, weighted covar for parameters
println("theta_stddev = ",theta_stddev)
end

if true
println("# Now resampling without emulator")
@time (theta_out, d_out) = ABC.generate_abc_sample(abc_plan, pop_out, ss_true, target_epsilon_full, num_plot=num_plot);
theta_mean = sum(theta_out,2)/size(theta_out,2)
println("theta_mean = ",theta_mean)
theta_stddev = sqrt(ABC.var_weighted(theta_out'.-theta_mean',ones(size(theta_out,2))))  # scaled, weighted covar for parameters
println("theta_stddev = ",theta_stddev)
end

quit()
##=
using PyPlot
eps_thresh = maximum(pop_out.dist)
idx_out = find(x->x<eps_thresh,pop_out.dist);
plot_abc_posterior(pop_out.theta[:,idx_out],1,"r-")
plot_abc_posterior(pop_out.theta[:,idx_out],2,"g-")
plot_abc_posterior(pop_out.theta[:,idx_out],3,"b-")
plot_abc_posterior(pop_out.theta[:,idx_out],4,"m-")
plot_abc_posterior(pop_out.theta[:,idx_out],5,"c-")

idx_emu = find(x->x<eps_thresh,pop_emu.dist)
plot_abc_posterior(pop_emu.theta[:,idx_emu],1,"r.")
plot_abc_posterior(pop_emu.theta[:,idx_emu],2,"g.")
plot_abc_posterior(pop_emu.theta[:,idx_emu],3,"b.")
plot_abc_posterior(pop_emu.theta[:,idx_emu],4,"m.")
plot_abc_posterior(pop_emu.theta[:,idx_emu],5,"c.")
###=#

#=
abc_plan_emu2 = deepcopy(abc_plan)
abc_plan_emu.use_emulator = false
abc_plan_emu.num_max_times = 1
abc_plan_emu.target_epsilon = target_epsilon_full
pop_emu2 = run_abc(abc_plan_emu2, ss_true, pop_emu2; verbose=true, print_every=1, ss_scale=ss_scale);

idx_emu2 = find(x->x<eps_thresh,pop_emu2.dist)
plot_abc_posterior(pop_emu2.theta[:,idx_emu2],1,"r+")
plot_abc_posterior(pop_emu2.theta[:,idx_emu2],2,"g+")
plot_abc_posterior(pop_emu2.theta[:,idx_emu2],3,"b+")
plot_abc_posterior(pop_emu2.theta[:,idx_emu2],4,"m+")
plot_abc_posterior(pop_emu2.theta[:,idx_emu2],5,"c+")
=#

num_plot = 100
(theta_out, d_out) = ABC.generate_abc_sample(abc_plan, pop_out, ss_true, eps_thresh, num_plot=num_plot);
(theta_emu, d_emu) = ABC.generate_abc_sample(abc_plan, pop_emu, ss_true, eps_thresh, num_plot=num_plot);
#(theta_emu2, d_emu2) = ABC.generate_abc_sample(abc_plan, pop_emu2, ss_true, eps_thresh, num_plot=num_plot);

plot_abc_posterior(theta_out,1,"r-")
plot_abc_posterior(theta_out,2,"g-")
plot_abc_posterior(theta_out,3,"b-")
plot_abc_posterior(theta_out,4,"m-")
plot_abc_posterior(theta_out,5,"c-")

#=
plot_abc_posterior(theta_emu2,1,"r.")
plot_abc_posterior(theta_emu2,2,"g.")
plot_abc_posterior(theta_emu2,3,"b.")
plot_abc_posterior(theta_emu2,4,"m.")
plot_abc_posterior(theta_emu2,5,"c.")
=#

plot_abc_posterior(theta_emu,1,"r+")
plot_abc_posterior(theta_emu,2,"g+")
plot_abc_posterior(theta_emu,3,"b+")
plot_abc_posterior(theta_emu,4,"m+")
plot_abc_posterior(theta_emu,5,"c+")



function emulate_ss(theta::Array{Float64,2})
   pred = ABC.predict_gp(emu,theta,sigmasq_obs=ss_scale) #,calc_sigma_pred=calc_stddev_summary_stats_mean_stddev_rate_period_snr_detected)
   #ABC.emulator_output_to_ss_mean(pred)
   ABC.emulator_output_to_ss_mean_stddev(pred)
  end
  function emulate_ss(theta::Array{Float64,1})
   pred = ABC.predict_gp(emu,reshape(theta,size(theta,1),1),sigmasq_obs=ss_scale) # ,calc_sigma_pred=calc_stddev_summary_stats_mean_stddev_rate_period_snr_detected)
   #ABC.emulator_output_to_ss_mean(pred)
   ABC.emulator_output_to_ss_mean_stddev(pred)
  end

j=k=1
dist_thresh = quantile(pop_out.reject_log.dist,0.25)
idx = sample(find(x->x<dist_thresh,pop_out.reject_log.dist),400);
plt[:errorbar]([pop_out.reject_log.theta[i][j] for i in idx],
               [emulate_ss(pop_out.reject_log.theta[i]).mean[k] for i in idx].-[pop_out.reject_log.ss[i].mean[k] for i in idx],
               yerr=[emulate_ss(pop_out.reject_log.theta[i]).stddev[k] for i in idx],fmt="r.")
#dist_thresh = quantile(pop_out.accept_log.dist,0.2)
idx = find(x->x<dist_thresh,pop_out.accept_log.dist);
plt[:errorbar]([pop_out.accept_log.theta[i][j] for i in idx],
               [emulate_ss(pop_out.accept_log.theta[i]).mean[k] for i in idx].-[pop_out.accept_log.ss[i].mean[k] for i in idx],
               yerr=[emulate_ss(pop_out.accept_log.theta[i]).stddev[k] for i in idx],fmt="b.")

dist_thresh = quantile(pop_out.reject_log.dist,1.0)
idx = sample(find(x->x<dist_thresh,pop_out.reject_log.dist),400);
plt[:errorbar]([pop_out.reject_log.theta[i][j] for i in idx],
               [emulate_ss(pop_out.reject_log.theta[i]).mean[k] for i in idx].-ss_true[k],
               yerr=[emulate_ss(pop_out.reject_log.theta[i]).stddev[k] for i in idx],fmt="r.")
#dist_thresh = quantile(pop_out.accept_log.dist,0.25)
idx = find(x->x<dist_thresh,pop_out.accept_log.dist);
plt[:errorbar]([pop_out.accept_log.theta[i][j] for i in idx],
               [emulate_ss(pop_out.accept_log.theta[i]).mean[k] for i in idx].-ss_true[k],
               yerr=[emulate_ss(pop_out.accept_log.theta[i]).stddev[k] for i in idx],fmt="b.")


#=
emu_out = emulate_ss(theta_true)
  abc_plan_emu = deepcopy(abc_plan)
  abc_plan_emu.emulator = emulate_ss
  abc_plan_emu.use_emulator = true
  abc_plan_emu.num_max_times = 50
  pop_emu = init_abc(abc_plan_emu,ss_true)

  pop_out = run_abc(abc_plan_emu, ss_true, pop_emu; verbose=true, print_every=1)


  abc_plan.num_max_times = 1
  pop_out2 = run_abc(abc_plan, ss_true, pop_out; verbose=true, print_every=1)
  pop_out = pop_out2

using JLD
save("ex4_out.jld", "pop_out", pop_out)
save("emu_out.jld", "pop_all", pop_all,"pop_snr",pop_snr,"pop_snr_geo",pop_snr_geo)

=#

#=
sampler_plot = abc_plan.make_proposal_dist(pop_out, 1.0) # abc_plan.tau_factor)
num_plot = 100
epsilon = 0.05
theta_plot, dist_plot = generate_abc_sample(sampler_plot,size(pop_out.theta, 1), ss_true, epsilon, num_plot=num_plot )
plot_abc_posterior(theta_plot,1)
med_dist = median(dist_plot)
plot_abc_posterior(theta_plot[:,find(dist_plot.<=med_dist)],1)
=#

