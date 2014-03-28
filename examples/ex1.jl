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



using PyPlot

plt.hist(pop_out.weights*length(pop_out.weights));
plt.hist(pop_out.dist);

num_param = 2
num_grid_x = 100
num_grid_y = 100
limit = 1.0
x = linspace(theta_true[1]-limit,theta_true[1]+limit,num_grid_x);
y = linspace(theta_true[2]-limit,theta_true[2]+limit,num_grid_y);
z = zeros(Float64,(num_param,length(x),length(y)))
for i in 1:length(x), j in 1:length(y) 
    z[1,i,j] = x[i]
    z[2,i,j] = y[j]
end
z = reshape(z,(num_param,length(x)*length(y)))
zz = [ ABC.pdf(ABC.GaussianMixtureModelCommonCovar(pop_out.theta,pop_out.weights,ABC.cov_weighted(pop_out.theta',pop_out.weights)),vec(z[:,i])) for i in 1:size(z,2) ]
zz = reshape(zz ,(length(x),length(y)));
levels = [exp(-0.5*i^2)/sqrt(2pi^num_param) for i in 0:5];
PyPlot.contour(x,y,zz',levels);
plot(pop_out.theta[1,:],pop_out.theta[2,:],".");

