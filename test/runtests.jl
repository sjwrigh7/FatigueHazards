cd(@__DIR__)
using Pkg
Pkg.activate("..")

using Revise
using Distributions
using LatinHypercubeSampling
using GaussianProcesses
#using Optim
#using ADTypes
using BlackBoxOptim

using FatigueHazards
#Pkg.develop(path="/home/stephenw/programming/FatigueHazards")
using Test

using Plots
using StatsBase
using Profile

########################
## define material model
s_min = 1000.0
s_max = 20_000.0
n_min = 1e3
n_max = 1e8

mat = FatigueHazards.BilinearMaterial(
    s_max,
    s_min,
    n_max,
    n_min
)

####################################
#### generate synthetic data

#s0 = [1000.0,3000.0,6000.0,10000.0]
#ds = [1000.0,2000.0,4000.0,6000.0]
#n0 = [1e3,1e5,1e7]
## test design
s0 = [1000.0,6000.0] # starting stress
ds = [2000.0,6000.0] # stress step
n0 = [1e4,1e6] # number of cycles per stress level
n_rep = 3 # number of i.i.d. samples per test design point

# material strength error
error_dist = Normal(0.0,10^(-1.5))

# construct design
initial_design = FatigueHazards.sweep_design(s0,ds,n0,n_rep)
# generate data
initial_data = FatigueHazards.simulate_step_stress(mat,initial_design,error_dist)

######################################
## generate splines
begin
    spline_order = 4
    n_int = 3
    spl = FatigueHazards.init(initial_data,spline_order,n_int)
end

######################################
##### set up for MCMC
## find approximate MLE values for beta and gamma parameters
opt_vals = FatigueHazards.opt_lik(initial_data,spl)

## define step size for MCMC
steps = [
    2.0, # I basis function coeffs
    3.2,
    4.0,
    3.7,
    4.0,
    4.7,
    3.0,
    0.5 # beta
]

##################################
##### mcmc sampling of beta and gamma
## initial mcmc samples of beta and gamma
samples = FatigueHazards.mcmc_baseline_splines(initial_data,spl,40_000,steps,opt_vals)

#single_time = 1.053091
#single_n = 10_000
#yielded_iid = Int(floor(single_n / lag))
## calculate necessary lag
n_burn = 2000
lag,acf_vals = FatigueHazards.find_lag(
    samples.gamma,
    samples.beta,
    n_burn;
    target=0.05,
    grid_size=2000,
    results=true
)

## optionally plot ACF results
let 
    p = plot()
    plot!(log.(10,acf_vals.lags),acf_vals.beta,label="beta")
    for i in axes(acf_vals.gamma,2)
        plot!(log.(10,acf_vals.lags),acf_vals.gamma[:,i],label="gamma $i")
    end
    title!("Posterior Autocorrelation")
    xlabel!("Log10 Lag")
    ylabel!("Autocorrelation")
    display(p)
end

##############################################
## optional memory efficient aggregation of lagged samples
n_target = 10_000
max_size = 50_000
n_burn = 2000

@time bulk_samples = FatigueHazards.bulk_mcmc_baseline_splines(
    initial_data,
    spl,
    n_target,
    steps,
    opt_vals,
    n_burn,
    lag;
    length_lim=max_size,
    multithread=true
)
#n_rep = Int(floor((n_target / yielded_iid) * (single_n / max_size)))
#est_time = (n_rep * single_time / 12 + single_time) * (max_size / single_n)
#multi_time = 3.272694
#non_multi_time = 7.529408
#################################
## eval optimal next test point
next_test_point = FatigueHazards.optimize_design(
    bulk_samples,
    initial_data,
    spl,
    30;
    s_min=1e3,
    s_max=2e4,
    ds_min=-1e4,
    ds_max=1e4,
    n_max = initial_data.t_max,
    #n_const=5e3,
    reduce=false,
)

let 
    p = plot()
    scatter!(
        next_test_point.x',
        label=["Starting Stress" "Stress Step" "Cycles Per Stress Level"]    
    )
    xlabel!("Optimization Iteration")
    ylabel!("Normalized Parameter Value")
end

let 
    p = plot()
    scatter!(next_test_point.y,label=false)
    xlabel!("Optimization Iteration")
    ylabel!("Shannon Information")
end

_,best_idx = findmax(next_test_point.y)
best_inp = next_test_point.x[:,best_idx]

test_point = FatigueHazards.StepStressTest(
    best_inp[1] * (2e4 - 1e3) + 1e3,
    best_inp[2] * (1e4 + 1e4) - 1e4,
    best_inp[3] * (initial_data.t_max - 1e3) + 1e3
)
#####################################
## test sequential design
n_opt = 30
design_points = FatigueHazards.StepStressTest[]
posterior_means = Array{Float64}(undef,n_opt,spl.params.num_basis + 1)
posterior_vars = similar(posterior_means)
curr_stresses = copy(initial_data.raw.stresses)
curr_cycles = copy(initial_data.raw.cycles)
for i in 1:n_opt
    new_data = FatigueHazards.simulate_step_stress(
        mat,
        [test_point],
        error_dist
    ).raw

    combined_stresses = vcat(curr_stresses,new_data.stresses)
    combined_cycles = vcat(curr_cycles,new_data.cycles)
    combined_data = FatigueHazards.StepStressRawData(
        combined_stresses,
        combined_cycles
    )
    full_data = FatigueHazards.partition_time(combined_data)
    ## generate splines
    spline_order = 4
    n_int = 3
    spl = FatigueHazards.init(full_data,spline_order,n_int)

    ## find approximate MLE values for beta and gamma parameters
    opt_vals = FatigueHazards.opt_lik(full_data,spl)
    ## initial mcmc samples of beta and gamma
    samples = FatigueHazards.mcmc_baseline_splines(full_data,spl,10_000,steps,opt_vals)

    ## calculate necessary lag
    n_burn = 2000
    lag,acf_vals = FatigueHazards.find_lag(
        samples.gamma,
        samples.beta,
        n_burn;
        target=0.05,
        grid_size=2000,
        results=true
    )

    bulk_samples = FatigueHazards.bulk_mcmc_baseline_splines(
        full_data,
        spl,
        n_target,
        steps,
        opt_vals,
        n_burn,
        lag;
        length_lim=max_size
    )
    
    posterior_means[i,1] = mean(bulk_samples.beta)
    posterior_means[i,2:end] = mean(bulk_samples.gamma,dims=1)

    posterior_vars[i,1] = var(bulk_samples.beta)
    posterior_vars[i,2:end] = var(bulk_samples.gamma,dims=1)

    test_point = FatigueHazards.optimize_design(
        bulk_samples,
        full_data,
        spl,
        30;
        s_min=1e3,
        s_max=2e4,
        ds_min=-1e4,
        ds_max=1e4,
        n_max = full_data.t_max,
        #n_const=5e3,
        reduce=false,
    )
    push!(design_points,test_point)
end
#################################
design_norm = FatigueHazards.StepStressTest(
    1e4 / initial_data.s_max,
    3e3 / initial_data.s_max,
    3e3 / initial_data.t_max
)

test1 = FatigueHazards.init_design(
    design_norm,
    spl.params.design
)


test_design = FatigueHazards.StepStressTest(
    1e4, #/ initial_data.s_max,
    3e3, #/ initial_data.s_max,
    3e3 #/ initial_data.t_max
)

test2 = Vector{Float64}(undef,100)
test_risk = [FatigueHazards.sum_risk(j,initial_data.s_norm,bulk_samples.beta[1],initial_data.in_risk_idx) for j in 2:(length(initial_data.t_norm)-1)]
@profview for i in 1:10000
    FatigueHazards.cumulative_hazard_scalar(
        view(test2,1:10),
        view(test_risk,2:11),
        view(test1[1].I_diff,1:10)
    )
end

n_sim = 10000
n_rep = 10
n_marg_evals = 2500
log_cond_vals3 = Array{Float64}(undef,n_sim,n_rep)
log_marg_vals3 = similar(log_cond_vals3)

@showprogress for i in 1:n_rep
    log_cond,log_marg_vec = FatigueHazards.eval_entropy(
        test_design,
        initial_data,
        bulk_samples,
        spl.params.design,
        n_sim,
        n_marg_evals;
        results=:vector,
        multithread=true
    )
    log_cond_vals3[:,i] .= log_cond
    log_marg_vals3[:,i] .= log_marg_vec
end

let 
    n = size(log_cond_vals,1)
    r_idx = sample(1:n,n,replace=false)
    p = plot()
    for i in 1:n_rep
        plot!(
            cumsum(log_cond_vals[r_idx,i]) ./ collect(1:n),
            label=false,
            alpha=0.7
        )
    end
    title!("MC Simulation Consistency")
    xlabel!("MC Iteration")
    ylabel!("MC Expectation")
    #ylims!((5,6))
    display(p)
end

let 
    m = size(log_marg_vals,2)
    n = size(log_marg_vals,1)
    outer_idx = [1,100,1000]#1:10:m
    #r_idx = sample(1:n,n,replace=false)
    p = plot()
    for j in 1:n_rep
        for i in outer_idx
            temp = exp.(log_marg_vals[i,:,j])

            plot!(
                log.(cumsum(temp) ./ collect(1:n)),
                label=false,
                alpha=0.7,
                color=i
            )
        end
    end
    title!("Consistency of MC")
    xlabel!("MC Iteration")
    ylabel!("MC Expectation")
    display(p)
end

let 
    n = size(log_marg_vals,1)
    r_idx = sample(1:n,n,replace=false)

    p = plot()
    for i in 1:n_rep
        plot!(
            cumsum(log_marg_vals[r_idx,i]) ./ collect(1:n),
            label=false,
            alpha=0.7
        )
    end
    title!("MC Consistency")
    xlabel!("MC Iteration")
    ylabel!("MC Expectation")
    #ylims!((4.7,5.4))
end

let 
    n = size(log_cond_vals,1)
    r_idx = sample(1:n,n,replace=false)
    p = plot()
    for i in 1:n_rep
        marg_term = cumsum(log_marg_vals[r_idx,i]) ./ collect(1:n)
        cond_term = cumsum(log_cond_vals[r_idx,i]) ./ collect(1:n)

        plot!(
            (cond_term .- marg_term),
            label=false,
            alpha=0.7
        )
    end
    title!("MC Consistency")
    xlabel!("MC Iteration")
    ylabel!("Shannon Information")
    display(p)
end

let 
    start = round(Int,0.5 * size(log_marg_full,1))
    start = 1
    idx = round.(
        Int,
        range(
            start = 1,
            stop = size(log_marg_full,2),
            length = 12
        )
        )
    p = plot()
    plot!(
        log_marg_full[start:end,idx],
        label = false    
    )
    xlabel!("MC Iteration")
    ylabel!("Log-Expectation of Marginal")
    title!("Convergence of Inner MC")
    #plot!(exp.(test3))
end

log_cond,log_marg_vec = FatigueHazards.eval_entropy(
    test_design,
    initial_data,
    bulk_samples,
    spl.params.design,
    3000,
    3000;
    results=:vector
)

test1 = (cumsum(log_cond) ./ collect(1:length(log_cond))) .- 
    (cumsum(log_marg_vec) ./ collect(1:length(log_marg_vec)))
noise_est = log(var(test1[2500:3000]))

test1 = FatigueHazards.eval_entropy(
    test_design,
    initial_data,
    bulk_samples,
    spl.params.design,
    3000,
    3000;
    results=:scalar
)
#FatigueHazards.geweke_statistic(log_marg_full[:,3000];burn=1200,norm=true)

#TODO: check consistency of entropy calculations for previous methods
#TODO... i.e., on master branch
let 
    r_idx = sample(
        1:length(mdl_eval[1]),
        length(mdl_eval[1]),
        replace=false
    )
    #FatigueHazards.geweke_statistic(log_marg_vec[r_idx];burn=2000)
    p = plot()
    plot!(
        cumsum(mdl_eval[1][r_idx]) ./ (collect(1:length(mdl_eval[1]))),
        label="Log-Conditional"
    )
    plot!(
        cumsum(mdl_eval[2][r_idx]) ./ (collect(1:length(mdl_eval[2]))),
        label="Log-Marginal"
    )
    xlabel!("MC Iteration")
    ylabel!("Expectation")
    title!("Convergence of Outer MC")
end

#########################
# bayesian optimization of test design
s_min = 1e3
s_max = 2e4

n_min = 1e3
n_max = 1e7
n_const = 5e3

ds_min = 1e1
ds_max = 1e4

doe_bounds = [
    (s_min,s_max),
    (ds_min,ds_max),
    #(n_min,n_max)
]
opt_bounds = [(0.0,1.0) for i in eachindex(doe_bounds)]

init_size = 15
n_rep = 10
doe = LHCoptim(init_size,2,100)
doe = scaleLHC(doe[1],doe_bounds)

lower_bounds = [p[1] for p in doe_bounds]
upper_bounds = [p[2] for p in doe_bounds]

doe_norm = (doe .- lower_bounds') ./ (upper_bounds' .- lower_bounds')

ent_vals = Array{Float64}(undef,init_size,n_rep)

begin
    @showprogress "Evaluating..." for i in 1:init_size
        temp_design = FatigueHazards.StepStressTest(
            doe[i,1],
            doe[i,2],
            n_const#doe[i,3]
        )

        for j in 1:n_rep
            ent_vals[i,j] = FatigueHazards.eval_entropy(
                temp_design,
                initial_data,
                bulk_samples,
                spl.params.design,
                5000,
                2500;
                results=:scalar,
                multithread=true
            )
        end
    end
end

bad_case = 3
n_sim = 5000
n_rep = 10
n_marg_evals = 2500
log_cond_vals = Array{Float64}(undef,n_sim,n_rep)
log_marg_vals = similar(log_cond_vals)

let
    temp_design = FatigueHazards.StepStressTest(
        doe[bad_case,1],
        doe[bad_case,2],
        n_const#doe[i,3]
    )

    @showprogress for i in 1:n_rep
        log_cond,log_marg_vec = FatigueHazards.eval_entropy(
            temp_design,
            initial_data,
            bulk_samples,
            spl.params.design,
            n_sim,
            n_marg_evals;
            results=:vector,
            multithread=true
        )
        log_cond_vals[:,i] .= log_cond
        log_marg_vals[:,i] .= log_marg_vec
    end
end

# initialize GP
mdl = ElasticGPE(
    length(doe_bounds),
    mean = MeanConst(0.5),
    kernel = SE(repeat([-1.50],length(doe_bounds)),-2.0),
    logNoise = -1.0,
    capacity = 3000
)

# set priors for GP
set_priors!(mdl.mean,[Normal(0.5,1.0)])
set_priors!(mdl.logNoise,[Normal(-1.0,1.0)])
set_priors!(mdl.kernel,vcat(repeat([Normal(-1.50,1.5)],length(doe_bounds)),Normal(-2.0,2.0)))

doe_long = repeat(doe_norm,inner)
append!(mdl,permutedims(doe_norm),(ent_vals))

try
    GaussianProcesses.optimize!(mdl,noise=false)
catch
    global chain = ess(mdl;nIter = 20000,noise=false)
end

#chain = ess(mdl;nIter=30000,noise=false)
let 
    plot(chain')
end
if length(doe_bounds) == 2
    let
        p = plot()
        plot(mdl,fill=true,label="GP",colorbartitle="Shannon Information")
        scatter!(doe_norm[:,1],doe_norm[:,2],zcolor=ent_vals,label="Data")
        xlabel!("Starting Stress")
        ylabel!("Stress Step")
    end
end

function objective(theta)
    mdl_out = predict_f(mdl,permutedims(theta'))
    upper_CI = mdl_out[1][1] + 1.645 * mdl_out[2][1]
    return -upper_CI
end

n_opt = 20
begin
    for i in 1:n_opt
        #x0 = [
        #    rand(Uniform(s_min,s_max)),
        #    rand(Uniform(ds_min,ds_max)),
        #    rand(Uniform(n_min,n_max))
        #]
        opt_res = bboptimize(
                objective;
                SearchRange = opt_bounds,
                PopulationSize=5_000,
                MaxTime = 1.5
            )
        norm_vals = best_candidate(opt_res)
        scaled_vals = norm_vals .* (upper_bounds .- lower_bounds) .+ lower_bounds

        temp_design = FatigueHazards.StepStressTest(
            scaled_vals[1],
            scaled_vals[2],
            n_const#opt_val[3],
        )
        mdl_eval = FatigueHazards.eval_entropy(
            temp_design,
            initial_data,
            bulk_samples,
            spl.params.design,
            2500,
            3000;
            results=:scalar,
            multithread=true
        )

        append!(mdl,permutedims(norm_vals'),[mdl_eval])
        chain = ess(mdl;nIter=30000,noise=true,domean=false)
        #GaussianProcesses.optimize!(mdl)
    end
end

# initialize GP
mdl = ElasticGPE(
    length(doe_bounds),
    mean = MeanConst(0.5),
    kernel = SE(repeat([-1.0],length(doe_bounds)),-2.0),
    logNoise = noise_est,
    capacity = 3000
)
#append!(mdl,permutedims(doe_norm),ent_vals)
# set priors for GP
set_priors!(mdl.mean,[Normal(0.5,1.0)])
set_priors!(mdl.logNoise,[Normal(noise_est,0.5)])
set_priors!(mdl.kernel,vcat(repeat([Normal(-1.0,1.0)],length(doe_bounds)),Normal(-2.0,1.0)))
#set_priors!(mdl.kernel,vcat(repeat([Uniform(-2.0,2.0)],length(doe_bounds)),Uniform(-2.0,2.0)))

append!(mdl,curr_design,curr_resp)

GaussianProcesses.optimize!(mdl,noise=false)
chain = ess(mdl;nIter=30000,noise=false)
curr_design = copy(mdl.x)
curr_resp = copy(mdl.y)


chain = ess(mdl;nIter=60000)
chain = GaussianProcesses.mcmc(mdl;nIter=10000,noise=false,ε=0.1)#0.00041)
GaussianProcesses.optimize!(mdl)
let
    noise = vec(chain[1,:])
    mu = vec(chain[2,:])
    l1 = vec((chain[3,:]))
    l2 = vec((chain[4,:]))
    sig2 = vec((chain[5,:]))

    p = plot()

    plot!(noise,label="Noise")
    plot!(mu,label="μ")
    plot!(l1,label="ℓ1")
    plot!(l2,label="ℓ2")
    plot!(sig2,label="σ^2")
end

if length(doe_bounds) == 2
    let
        p = plot()
        #plot(mdl,fill=true,label="GP",colorbartitle="Shannon Information")
        scatter!(mdl.x[1,:],mdl.x[2,:],zcolor=mdl.y,label="Data")
        xlabel!("Starting Stress")
        ylabel!("Stress Step")
    end
end

#=
test1 = FatigueHazards._log_sum_exp(log_marg[:,1]) .- log.(collect(1:size(log_marg,1)))
est2 = log.(cumsum(exp.(log_marg[:,1])) ./ (collect(1:size(log_marg,1))))

test3 = log.(cumsum(exp.(log_marg[:,end])) ./ (collect(1:size(log_marg,1))))

maximum(log_marg[:,8108])
minimum(log_marg[:,8108])

test1 = FatigueHazards.eval_log_marg(log_marg;res=:full)

let 
    idx = 2497
    p = plot()
    plot!(exp.(test1[3000:end,(idx)]))
    #plot!(exp.(test3))
end

FatigueHazards.eval_inner_chain(log_marg;band=100)
=#


let 
    r_idx = sample(1:length(log_joint),length(log_joint),replace=false)
    joint = exp.(log_joint)
    marg = vec(mean(exp.(log_marg),dims=1))
    log_marg_vec = log.(marg)

    log_joint_running = cumsum(log_joint[r_idx]) ./ (1:length(log_joint))
    log_marg_running = cumsum(log_marg_vec[r_idx]) ./ (1:length(log_marg_vec))

    p1 = plot()
    plot!(log_joint_running,label="joint")
    plot!(log_marg_running,label="marginal")


end
#=
max_lag = 1000

lag_vals = round.(Int,10 .^ collect(
    range(
        start=0,
        stop=log(10,length(samples[2])/5),
        length=1000
    )
))
lag_vals = sort(unique(lag_vals))

ac_vals = hcat(autocor(samples[2],lag_vals),autocor(samples[1],lag_vals))
let 
    p = plot()
    plot!(log.(10,lag_vals),ac_vals[:,1],label="Beta")
    for i in 2:size(ac_vals,2)
        plot!(log.(10,lag_vals),ac_vals[:,i],label="Gamma $(i-1)")
    end
    title!("Posterior Samples Autocorrelation")
    xlabel!("Log10-Lag")
    ylabel!("ρ")
    display(p)
    savefig(p,save_path*"autocorr.png")
end
=#
gamma_use = vec(mean(gamma_samples,dims=1))
beta_use = mean(beta_samples)

risk_terms = exp.(beta_use * test1[2][2:end])

n_sample = 10_000
t_samps = Vector{Float64}(undef,n_sample)
k_vals = Vector{Int}(undef,n_sample)

for i in 1:n_sample
    risk_terms = exp.(beta_samples[i] * test1[2][2:end])
    t,k = FatigueHazards.sample_t(
        gamma_samples[i,:],
        test1[1],
        risk_terms,
        test1[3]
    )
    t_samps[i] = t
    k_vals[i] = k
end

histogram(t_samps)

#=
t_grid = collect(range(
    start = 0.0,
    stop = 0.01,
    step = 1e-7
))

stress_grid = Vector{Float64}(undef,length(t_grid))
stress_grid[1] = 0.0

target_t = design_norm.n
counter = 1
for i in 2:length(t_grid)
    divis = t_grid[i] / (counter * target_t)
    if divis >= 1.0
        counter += 1
    end
    stress_grid[i] = design_norm.s0  + (counter - 1) * design_norm.ds
end

u_samps = rand(Uniform(0.0,1.0),15000)

t_idx = Vector{Int}(undef,length(u_samps))
t_samps = Vector{Float64}(undef,length(u_samps))

risk_terms = exp.(beta_use * stress_grid[2:end])
splines = FatigueHazards.generate_splines(spline_order,spl.params.design.interior_knots,t_grid)
I_diff = splines.I_diff * gamma_use
M = splines.M * gamma_use

c_hazard = cumsum(I_diff .* risk_terms)
surv = exp.(-c_hazard)

for i in eachindex(u_samps)
    idx = findfirst(x -> x < u_samps[i],surv)
    t_idx[i] = idx
    t_samps[i] = t_grid[idx]
end

haz = M[2:end] .* (risk_terms)
#haz = min.(haz,1e300)

dens = surv .* haz

l_haz = log.(haz)

test_man = -c_hazard .+ l_haz

test_l = log.(dens)
test_l = max.(test_l,-log(prevfloat(floatmax(Float64))))

design_norm = FatigueHazards.StepStressTest(
    3e3 / initial_data.s_max,
    3e3 / initial_data.s_max,
    3e3 / initial_data.t_max
)

test1 = FatigueHazards.init_design(
    design_norm,
    spl.params.design
)

beta_burn = samples[2][(3000 + 1):end]
gamma_burn = samples[1][(3000 + 1):end,:]

beta_thin = beta_burn[1:10:end]
gamma_thin = gamma_burn[1:10:end,:]

r_idx = rand(1:length(beta_burn),500)

r_beta = beta_burn[r_idx]
r_gamma = gamma_burn[r_idx,:]

t_samples = Vector{Float64}(undef,500)
ks = Vector{Int}(undef,500)

for i in 1:500
    risk_terms = exp.(test1[2][2:end] .* r_beta[i])
    t,k = FatigueHazards.sample_t(r_gamma[i,:],test1[1],risk_terms,test1[3],1e-6)
    t_samples[i] = t
    ks[i] = k
end
=#
new_stress = Vector{Float64}(undef,length(k_vals))
for i in eachindex(new_stress)
    new_stress[i] = test1[2][k_vals[i] + 1]
end

merged_time = vcat(t_samps,test1[3])
merged_stress = vcat(new_stress,test1[2])

sort_idx = sortperm(merged_time)

fail_sort = findall(x -> in(x,1:length(t_samps)),sort_idx)
fail_idx = sort_idx[fail_sort]

combined_time,combined_stress,merged_idx,new_idx = FatigueHazards.merge_grids(test1[3],test1[2],t_samps,k_vals)


risk_random = [exp.(combined_stress * beta_samples[i]) for i in 1:n_sample]
I_diff_random = [test1[1].I_diff * gamma_samples[i,:] for i in 1:n_sample]
M_random = [test1[1].M * gamma_samples[i,:] for i in 1:n_sample]

risk_post = [exp.(combined_stress * beta_samples[j]) for j in 1:1000]
I_diff_post = [test1[1].I_diff * gamma_samples[j,:] for j in 1:1000]
M_post  = [test1[1].M * gamma_samples[j,:] for j in 1:1000]

log_dens = Vector{Float64}(undef,n_sample)
log_marg = Array{Float64}(undef,n_sample,length(risk_post))

for i in 1:n_sample
    log_dens[i] = FatigueHazards.log_density(
        t_samples[i],
        combined_time,
        risk_random[i],
        I_diff_random[i],
        M_random[i]
    )
    for j in eachindex(risk_post)
        log_marg[i,j] = FatigueHazards.log_density(
            t_samples[i],
            combined_time,
            risk_post[j],
            I_diff_post[j],
            M_post[j]
        )
    end
end



@profview FatigueHazards.eval_entropy(
    test_design,
    initial_data,
    500,
    10,
    3000,
    samples[2],
    samples[1],
    spl.params.design
)

log_dens,log_marg,combined_time,t_samples = FatigueHazards.eval_entropy(
    test_design,
    initial_data,
    100,
    50,
    100,
    3000,
    samples[2],
    samples[1],
    spl.params.design
)

log_dens,log_marg,combined_time,t_samples = FatigueHazards.eval_entropy(
    test_design,
    initial_data,
    50_000,#5000,
    1,#max_lag,
    0,#3000,
    beta_samples,
    gamma_samples,
    spl.params.design;
    n_thin = 50
)

reg_marg = exp.(log_marg)


let 
    p = plot()
    idx = 8999
    marg_running = (cumsum(reg_marg[:,idx]) ./ (1:size(log_marg,1)))
    #test1 = cumsum(log_marg[:,idx]) ./ (1:size(log_marg,1))
    plot!(marg_running[1000:end],label="Marginal")
    #plot!(test1)
    title!("Inner Monte Carlo Expectation")
    xlabel!("Iteration")
    ylabel!("Sample Mean")
    display(p)
    #savefig(p,save_path*"marginal_inner_mc.png")
end

marg_use = vec(log.(mean(reg_marg,dims=1)))

#marg_running = cumsum(vec(mean(log_marg,dims=1))) ./ (1:length(log_dens))
dens_running = cumsum(log_dens) ./ (1:length(log_dens))
marg_running = cumsum(marg_use) ./ (1:length(log_dens))

n_sim = 25
n_batch = 50
batch_size = 5000

full_joint = Array{Float64}(undef,n_batch*batch_size,n_sim)
full_marg = similar(full_joint)

for j in 1:n_sim
    for i in 1:n_batch
        res = FatigueHazards.eval_entropy(
            test_design,
            initial_data,
            5000,
            50,
            3000,
            samples[2],
            samples[1],
            spl.params.design
        )
        start_idx = batch_size * (i - 1) + 1
        stop_idx = batch_size * i

        full_joint[start_idx:stop_idx,j] = res[1]
        full_marg[start_idx:stop_idx,j] = vec(mean(res[2],dims=1))
    end
end

res = [FatigueHazards.eval_entropy(
    test_design,
    initial_data,
    5000,
    50,
    3000,
    samples[2],
    samples[1],
    spl.params.design
) for i in 1:50]

combined_joint = [res[i][1] for i in eachindex(res)]
combined_marg = [vec(mean(res[i][2],dims=1)) for i in eachindex(res)]

full_joint = reduce(vcat,combined_joint)
full_marg = reduce(vcat,combined_marg)

combined_t = [res[i][4] for i in eachindex(res)]
full_t = reduce(vcat,combined_t)

marg_running = cumsum(full_marg,dims=1) ./ (1:size(full_joint,1))
dens_running = cumsum(full_joint,dims=1) ./ (1:size(full_joint,1))

#marg_running = [mean(full_marg[1:i]) for i in eachindex(full_marg)]
#dens_running = [mean(full_joint[1:i]) for i in eachindex(full_joint)]

save_path = "/home/stephenw/Nextcloud/engr/PhD/fatigue/src/figures/"
let
    r_idx = sample(1:length(log_dens),length(log_dens),replace=false)
    #r_idx = 1:length(log_dens)
    
    dens_running = cumsum(log_dens[r_idx]) ./ (1:length(log_dens))
    marg_running = cumsum(marg_use[r_idx]) ./ (1:length(log_dens))
    p = plot()
    
    plot!(1:size(dens_running,1),(dens_running),label="Log-Joint",color=1)
    plot!(1:size(dens_running,1),(marg_running),label="Log-Marginal",color=2)
    #plot!(1:size(dens_running,1),exp.(dens_running[:,2:end]),label=false,color=1,lw=4)
    #plot!(1:size(dens_running,1),exp.(marg_running[:,2:end]),label=false,color=2)
    title!("Outer Monte Carlo Expectations")
    ylabel!("Sample Mean")
    xlabel!("Iteration")
    #ylims!((-1e24,20))
    display(p)
    savefig(p,save_path*"outer_mcmc_random.png")
end

let 
    p = histogram(full_t,normalize=true,label=false)
    title!("Failure Time Histogram")
    xlabel!("Sampled Failure Time")
    ylabel!("Normalized Count")
    xlims!((0.0,0.026))
    #savefig(p,save_path*"t_hist.png")
end

marg = exp.(log_marg)
time_norm = (maximum(t_samples) - minimum(t_samples)) / length(t_samples)

marg_norm = vec(mean(marg,dims=1)) / (sum(mean(marg,dims=1)) * time_norm)
log_marg_norm = log.(marg_norm)

dens1 = exp.(log_dens)
dens_norm = dens1 / (sum(dens1) * time_norm)
log_dens_norm = log.(dens_norm)

mean(log_marg_norm) - mean(log_dens_norm)
#TODO: look into optimizing entropy calculations

@testset "FatigueHazards.jl" begin
    # Write your tests here.
end
