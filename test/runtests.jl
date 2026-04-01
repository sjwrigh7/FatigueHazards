cd(@__DIR__)
using Pkg
Pkg.activate("..")

using Revise
using Distributions
using LatinHypercubeSampling
using GaussianProcesses
using Optim
using ADTypes
using BlackBoxOptim

using FatigueHazards
#Pkg.develop(path="/home/stephenw/programming/FatigueHazards")
using Test

using Plots
using StatsBase

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
samples = FatigueHazards.mcmc_baseline_splines(initial_data,spl,10_000,steps,opt_vals)

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
n_target = 3_000
max_size = 1_000_000

bulk_samples = FatigueHazards.bulk_mcmc_baseline_splines(
    initial_data,
    spl,
    n_target,
    steps,
    opt_vals,
    n_burn,
    lag;
    length_lim=max_size
)


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

log_cond,log_marg_full = FatigueHazards.eval_entropy(
    test_design,
    initial_data,
    bulk_samples,
    spl.params.design,
    3000,
    3000;
    results=:full
)

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
    plot!(log_marg_full[start:end,idx])
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

let 
    r_idx = sample(
        1:length(log_cond),
        length(log_cond),
        replace=false
    )
    FatigueHazards.geweke_statistic(log_marg_vec[r_idx];burn=2000)
    p = plot()
    plot!(
        cumsum(log_cond[r_idx]) ./ (collect(1:length(log_cond))),
        label="Log-Conditional"
    )
    plot!(
        cumsum(log_marg_vec[r_idx]) ./ (collect(1:length(log_marg_vec))),
        label="Log-Marginal"
    )
end

#########################
# bayesian optimization of test design
s_min = 1e3
s_max = 2e4

n_min = 1e3
n_max = 1e7
n_const = 5e3

ds_min = -1e4
ds_max = 1e4

doe_bounds = [
    (s_min,s_max),
    (ds_min,ds_max),
    #(n_min,n_max)
]

doe = LHCoptim(30,2,100)
doe = scaleLHC(doe[1],doe_bounds)

ent_vals = Vector{Float64}(undef,size(doe,1))
begin
    for i in axes(doe,1)
        temp_design = FatigueHazards.StepStressTest(
            doe[i,1],
            doe[i,2],
            n_const#doe[i,3]
        )


        ent_vals[i] = FatigueHazards.eval_entropy(
            temp_design,
            initial_data,
            bulk_samples,
            spl.params.design,
            2500,
            10000;
            results=:scalar
        )
    end
end

# initialize GP
mdl = ElasticGPE(
    length(doe_bounds),
    mean = MeanConst(0.0),
    kernel = SEArd(repeat([-1.0],length(doe_bounds)),-1.0),
    logNoise = log(1e-3),
    capacity = 3000
)

# set priors for GP
set_priors!(mdl.mean,[Normal(0.0,2.0)])
set_priors!(mdl.logNoise,[Normal(log(1e-3),0.1)])
set_priors!(mdl.kernel,vcat(repeat([Normal(1.0,10.0)],length(doe_bounds)),Normal(-1.0,2.0)))

append!(mdl,permutedims(doe),(ent_vals))


optimize!(mdl,noise=false)

chain = ess(mdl;nIter=30000,noise=false)

if length(doe_bounds) == 2
    let
        p = plot()
        plot(mdl,fill=true,label="GP",colorbartitle="Shannon Information")
        scatter!(doe[:,1],doe[:,2],zcolor=ent_vals,label="Data")
        xlabel!("Starting Stress")
        ylabel!("Stress Step")
    end
end

function objective(theta)
    mdl_out = predict_f(mdl,permutedims(theta'))
    upper_CI = mdl_out[1][1] + 1.65 * mdl_out[2][1]
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
                SearchRange = doe_bounds,
                PopulationSize=5_000,
                MaxTime = 1.5
            )
        opt_val = best_candidate(opt_res)

        temp_design = FatigueHazards.StepStressTest(
            opt_val[1],
            opt_val[2],
            n_const#opt_val[3],
        )
        mdl_eval = FatigueHazards.eval_entropy(
            temp_design,
            initial_data,
            bulk_samples,
            spl.params.design,
            3000,
            3000;
            results=:scalar
        )

        append!(mdl,permutedims(opt_val'),[mdl_eval])
        chain = ess(mdl;nIter=30000,noise=false)
    end
end

curr_design = copy(mdl.x)
curr_resp = copy(mdl.y)
append!(mdl,curr_design,curr_resp)

chain = ess(mdl;nIter=60000,noise=false)
chain = GaussianProcesses.mcmc(mdl;nIter=30000,noise=false)
GaussianProcesses.optimize!(mdl;noise=false)
let
    mu = vec(chain[1,:])
    l1 = vec((chain[2,:]))
    l2 = vec((chain[3,:]))
    sig2 = vec((chain[4,:]))

    p = plot()

    plot!(mu,label="μ")
    plot!(l1,label="ℓ1")
    plot!(l2,label="ℓ2")
    plot!(sig2,label="σ^2")
end

if length(doe_bounds) == 2
    let
        p = plot()
        plot(mdl,fill=true,label="GP",colorbartitle="Shannon Information")
        scatter!(doe[:,1],doe[:,2],zcolor=ent_vals,label="Data")
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
