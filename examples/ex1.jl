using ABC
using Distributions

num_data_default = 100
gen_data_normal(theta::Array, n::Integer = num_data_default) = rand(Normal(theta[1],theta[2]),num_data_default)

normalize_theta2_pos(theta::Array) =  theta[2] = abs(theta[2])
is_valid_theta2_pos(theta::Array) =  theta[2]>0.0 ? true : false

theta_true = [0.0, 1.0]
param_prior = Distributions.DiagNormal(theta_true,ones(length(theta_true)))

abc_plan = abc_pmc_plan_type(gen_data_normal,ABC.calc_summary_stats_mean_var,ABC.calc_dist_max, param_prior; is_valid=is_valid_theta2_pos,num_max_attempt=10000);

num_param = 2
data_true = abc_plan.gen_data(theta_true)
ss_true = abc_plan.calc_summary_stats(data_true)
#println("theta= ",theta_true," ss= ",ss_true, " d= ", 0.)

pop_out = run_abc(abc_plan,ss_true;verbose=true);



