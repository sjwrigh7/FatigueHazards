cd(@__DIR__)
#using Pkg
#Pkg.activate("..")

using Revise
using Distributions
using LatinHypercubeSampling
using GaussianProcesses
import Optim
import ADTypes
using BlackBoxOptim
using ProgressMeter

using FatigueHazards
#Pkg.develop(path="/home/stephenw/programming/FatigueHazards")
#using Test
using DelimitedFiles
using Plots
using StatsPlots
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

damage_rule = FatigueHazards.LinearDamage(
    0.0,
    1.0,
    0.0
)

####################################
#### generate synthetic data
test_constraints = FatigueHazards.TestConstraints(
    1000.0,
    20000.0
)
#s0 = [1000.0,2000.0,4000.0,6000.0,8000.0,10000.0,14000.0]
#ds = [0.0,100.0,1000.0,2000.0,4000.0,6000.0]
#n0 = [1e3,1e4,1e5,1e6,1e7]
## test design
s0 = [1000.0,6000.0] # starting stress
ds = [2000.0,6000.0] # stress step
n0 = [1e4,1e6] # number of cycles per stress level

#s0 = exp.(collect(
#    range(
#        start = log(1000.0),
#        stop=log(20000.0),
#        length = 150
#    )
#))
#ds = [0.0]
#n0 = [1e10]

n_rep = 3 # number of i.i.d. samples per test design point

# material strength error
error_dist = Normal(0.0,10^(-1.5))

# construct design
initial_design = FatigueHazards.sweep_design(s0,ds,n0,n_rep)

# load existing data
optimized_designs = readdlm("../examples/sequential_design-fixed-designs.txt",',')

for i in axes(optimized_designs,1)
    push!(
        initial_design,
        FatigueHazards.StepStressTest(
            optimized_designs[i,1],
            optimized_designs[i,2],
            optimized_designs[i,3]
        )
    )
end
# generate data
initial_data = FatigueHazards.simulate_step_stress(
    damage_rule,
    mat,
    initial_design,
    error_dist,
    test_constraints
)

#=
added_design = repeat([design_points[1]],3)
new_data = FatigueHazards.simulate_step_stress(
    damage_rule,
    mat,
    added_design,
    error_dist,
    test_constraints
)
total_cycles = vcat(
    initial_data.raw.cycles,
    new_data.raw.cycles
)
total_stresses = vcat(
    initial_data.raw.stresses,
    new_data.raw.stresses
)
total_data = FatigueHazards.partition_time(
    FatigueHazards.StepStressRawData(
        total_stresses,
        total_cycles
    )
)
total_data.t_norm[2:(end-1)] .= total_data.t_norm[2:(end-1)] .* total_data.t_max / initial_data.t_max
total_data.s_norm[2:(end-1),:] .= total_data.s_norm[2:(end-1),:] .* total_data.s_max / initial_data.s_max
=#

#initial_data.t_norm[2:(end-1)] = log.(initial_data.t_norm[2:(end-1)] * initial_data.t_max) #./ log(initial_data.t_max)
#initial_data.s_norm[2:(end-1),:] = log.(initial_data.s_norm[2:(end-1),:] * initial_data.s_max) #./ log(initial_data.s_max)

#initial_data.t_norm[2:(end-1)] .= initial_data.t_norm[2:(end-1)] .* initial_data.t_max ./ real_t_max
#initial_data.s_norm[2:(end-1),:] .= initial_data.s_norm[2:(end-1),:] .* initial_data.s_max ./ real_s_max

######################################
## generate splines
begin
    base_haz_spline_order = 3
    base_haz_n_int = 2
    risk_spline_order = 3
    risk_n_int = 2
    base_haz_spl,risk_spl = FatigueHazards.init(initial_data,base_haz_spline_order,base_haz_n_int,risk_spline_order,risk_n_int)
    s_map = FatigueHazards.map_unique(initial_data)
    s_map[2:(end-1),:] .-= 1
    #base_haz_spl = FatigueHazards.init(initial_data,base_haz_spline_order,base_haz_n_int)
end

general_t_grid = exp.(
    collect(
        range(
            start = log.(100.0 / initial_data.t_max),
            stop = log.(base_haz_spl.params.knot_grid[end] - sqrt(eps(Float64))),
            length = 1000
        )
    )
)

general_s_grid = exp.(
    collect(
        range(
            start = log(s_min / initial_data.s_max),
            stop = log(s_max / initial_data.s_max),
            length = 150
        )
    )
)

function plot_ent(ent_res)
    log_cond = ent_res[1]
    log_marg = ent_res[2]

    n = length(log_cond)
    r_idx = sample(1:n,n,replace=false)
    p = plot()
    plot!(cumsum(log_cond[r_idx]) ./ collect(1:n),label="Conditional")
    plot!(cumsum(log_marg[r_idx]) ./ collect(1:n),label="Marginal")
    xlabel!("Iter")
    ylabel!("Sample Expectation")
end
#FatigueHazards.update_x!(risk_spl,sort(unique(total_data.s_norm)))
#FatigueHazards.update_x!(base_haz_spl,total_data.t_norm[1:(end-1)])
#s_map = FatigueHazards.map_unique(total_data)
#s_map[2:(end-1),:] .-= 1
#begin
#    FatigueHazards.update_x!(base_haz_spl,initial_data.t_norm[1:(end-1)])
#    FatigueHazards.update_x!(risk_spl,sort(unique(initial_data.s_norm)))
#    s_map = FatigueHazards.map_unique(initial_data)
#end

########################
## Run MCMC for spline risk function
begin
    ## find approximate MLE values for beta and gamma parameters
    #opt_vals = FatigueHazards.opt_lik(
    #    initial_data,
    #    base_haz_spl,
    #    risk_spl,
    #    s_map
    #)
    #for i in eachindex(opt_vals)
    #    opt_vals[i] = max(opt_vals[i],1e-50)
    #end
    ## initial mcmc samples of beta and gamma
    steps,init_vals,test_beta,test_gamma = FatigueHazards.find_stepsize(
        #total_data,
        initial_data,
        base_haz_spl,
        risk_spl,
        400,
        70,
        s_map;
        target=[0.44,0.44],
        #init_vals = opt_vals,
        init_vals=repeat(
            [1.0],
            base_haz_spl.params.num_basis + risk_spl.params.num_basis
        ),
        make_plots=true,
        show_plots=true,
        save_plots=false,
        init=0.001,
        scale = 0.7,
        shape=5.0,
        offset=0.0
    )
    for i in eachindex(steps.beta)
        steps.beta[i] = min(steps.beta[i],1.0)
    end
    for i in eachindex(steps.gamma)
        steps.gamma[i] = min(steps.gamma[i],1.0)
    end

    samples = FatigueHazards.mcmc_risk_splines(
        #total_data,
        initial_data,
        base_haz_spl,
        risk_spl,
        20_000,
        steps,
        s_map,
        init_vals
    )
end
begin
    n_burn = 1000
    lag,acf_vals = FatigueHazards.find_lag(
        samples.gamma,
        samples.beta,
        n_burn;
        target=0.1,
        grid_size=2000,
        results=true
    )
    lag = min(lag,1500)

    n_target = 2500
    max_size = 20_000

    @time bulk_samples = FatigueHazards.bulk_mcmc_risk_splines(
        #total_data,
        initial_data,
        base_haz_spl,
        risk_spl,
        s_map,
        n_target,
        steps,
        init_vals,
        n_burn,
        lag;
        length_lim=max_size,
        multithread=true
    )
end

case = "initial_data-logstress_design-spline_order_3-inter_knots_2-linstress_model"
writedlm("results/posterior_samples-$case.txt",hcat(bulk_samples.beta,bulk_samples.gamma))

#samples_use = readdlm("results/sequential_design_post.txt")
#beta_samples = samples_use[:,1:4]
#gamma_samples = samples_use[:,5:end]

# Posterior predictions of failure time
begin
    #=
    n_sample = 1000
    stress_design = exp.(
        collect(
            range(
                start = log.(minimum(initial_data.s_norm[2,:])),
                stop = log.(maximum(initial_data.s_norm)),
                length = 150
            )
        )
    )
    stress_design[1] += sqrt(eps(Float64))
    t_samples = Array{Float64}(undef,n_sample,length(stress_design))
    t_grid = vcat(
        0.0,
        (
            collect(
                range(
                    start = initial_data.t_norm[2],
                    stop = base_haz_spl.params.knot_grid[end] - sqrt(eps(Float64)),
                    length = 1000
                )
            )
        ) 
    )
    =#
    FatigueHazards.update_x!(risk_spl,general_s_grid)
    FatigueHazards.update_x!(base_haz_spl,general_t_grid)
    for i in eachindex(general_s_grid)
        
        M_beta = repeat(risk_spl.M[i,:]',length(general_t_grid)-1)
        
        for j in 1:n_sample
            beta_use = bulk_samples.beta[j,:]
            gamma_use = bulk_samples.gamma[j,:]
            
            # pre calculate risk terms over time grid
            risk_terms = exp.(M_beta * beta_use)
            
            t,_ = FatigueHazards.sample_t(
                gamma_use,
                base_haz_spl,
                risk_terms,
                general_t_grid,
                1e-6
            )
            t_samples[j,i] = t
        end
    end
end

writedlm("results/time_samples-$case.txt",t_samples .* initial_data.t_max)
writedlm("results/stress_grid-$case.txt",stress_design .* initial_data.s_max)

let 
    t_vals = Vector{Float64}(undef,1000)
    t_vals2 = similar(t_vals)
    beta_use = vec(mean(bulk_samples.beta,dims=1))
    gamma_use = vec(mean(bulk_samples.gamma,dims=1))

    M_beta = repeat(risk_spl.M[end,:]',length(general_t_grid)-1)
    risk_terms = exp.(M_beta * beta_use)

    cumu_base = base_haz_spl.I * gamma_use

    survival = exp.(- cumu_base .* risk_terms[1])
    
    cdf_vals = 1 .- survival
    println(cdf_vals)

    for i in eachindex(t_vals)
        u = rand(Uniform(0.0,1.0))
        if cdf_vals[end] < u
            idx = length(t_vals)
        else
            idx = findfirst(x -> x >= u,cdf_vals)
        end
        t_vals[i] = general_t_grid[idx]
        
        t,_ = FatigueHazards.sample_t(
            gamma_use,
            base_haz_spl,
            risk_terms,
            general_t_grid,
            1e-6
        )
        t_vals2[i] = t
    end
    t_vals .+= sqrt(eps(Float64))
    t_vals2 .+= sqrt(eps(Float64))

    histogram(log.(10,t_vals2.* initial_data.t_max),alpha=0.2)
    histogram!(log.(10,t_vals.* initial_data.t_max),alpha=0.2)
    xlims!((1.0,5.0))
end

# plot posterior prediction of failure time
let 
    log_s = log.(10,general_s_grid .* initial_data.s_max)
    log_n = log.(10,t_samples .* initial_data.t_max)

    truth_failure_time = Vector{Float64}(undef,length(log_s))
    for i in eachindex(log_s)
        truth_failure_time[i] = FatigueHazards.bilinear_sn(
            mat,
            exp(
                log_s[i] * log(10.0)
            )
        )
    end

    p = plot()
    # error line of model predicted failure time
    errorline!(
        log_s,
        log_n,
        label="Estimate",
    )
    # subtle lines to at observed failure times
    hline!(
        log.(10,initial_data.t_norm .* initial_data.t_max),
        label="Observed Failure Times",
        alpha=0.2
    )
    # plot true failure times
    plot!(
        log_s,
        log.(10,truth_failure_time),
        label="Truth"
    )
    xlabel!("Log-10 Stress")
    ylabel!("Log-10 Failure Time")
    title!("Model Estimate v.s. Truth")
    #savefig(p,"debug/fixed_fit.png")
end

# plot posterior estimate of risk splines
let 
    risk_vals = risk_spl.M[2:end,:] * samples.beta[10001:11000,:]'

    s_vals = log.(10,stress_design .* initial_data.s_max)
    p = plot()
    errorline!(s_vals[2:end],risk_vals,label="Estimate")
    knot_vals = log.(10,risk_spl.params.knot_grid .* initial_data.s_max)
    vline!(knot_vals,label="Knots")
    xlabel!("Stress")
    ylabel!("Value")
    title!("Posterior Estimate of Stress Spline")
    #savefig(p,"debug/stress_spline1.png")
end

# plot posterior estimate of baseline hazard splines
let 
    haz_vals = base_haz_spl.M * samples.gamma[10001:11000,:]'

    t_vals = log.(10,t_grid .* initial_data.t_max)
    p = plot()
    errorline!(t_vals,haz_vals,label="Posterior Estimate")
    knot_vals = log.(10,base_haz_spl.params.knot_grid .* initial_data.t_max)
    knot_vals[knot_vals .== -Inf] .= 0.0
    vline!(knot_vals,label="Knots")
    vline!(log.(10,initial_data.t_norm[1:(end-1)] .* initial_data.t_max),label=false,alpha=0.2)
    xlabel!("Log-10 Cycles")
    ylabel!("Value")
    title!("Posterior Estimate of Baseline Hazard Spline")
    #savefig(p,"debug/hazard_spline2.png")
end

# find new test point
test_point = FatigueHazards.optimize_design(
    bulk_samples,
    initial_data,
    base_haz_spl,
    risk_spl,
    10;
    s_min=1e3,
    s_max=2e4,
    ds_min=-1e4,
    ds_max=1e4,
    n_max = (base_haz_spl.params.knot_grid[end] - sqrt(eps(Float64))) * initial_data.t_max,
    n_min = 1000.0,
    n_const=5e3,
    n_mcmc=5000,
    n_init=15,
    n_use=3,
    n_rep=3,
    reduce=false,
)

temp_design = FatigueHazards.StepStressTest(
    16000.0,
    -400.0,
    1000.0
)
temp_ent = FatigueHazards.eval_entropy(
    temp_design,
    initial_data,
    bulk_samples,
    base_haz_spl,
    risk_spl,
    2500,
    2500;
    results=:vector,
    multithread=true,
    return_times=true
)

#####################################
# run sequential design
case = "sequential_design-fixed-"
n_t_sample = 1000
n_posterior_save = 1000
n_rep_obs = 3

ds_min = -10000.0
ds_max = 10000.0
n_const = 5e3
n_opt = 15

design_points = FatigueHazards.StepStressTest[]
#beta_draws = Array{Float64}(undef,1000,size(bulk_samples.beta,2),n_opt)
#gamma_draws = Array{Float64}(undef,1000,size(bulk_samples.gamma,2),n_opt)
curr_stresses = copy(initial_data.raw.stresses)
curr_cycles = copy(initial_data.raw.cycles)
combined_stresses = copy(initial_data.raw.stresses)
combined_cycles = copy(initial_data.raw.cycles)
t_samples = Array{Float64}(undef,n_t_sample,length(general_s_grid))
#test_point = FatigueHazards.StepStressTest(
#    1.0 * s_max,
#    0.643 * (ds_max - ds_min) + ds_min,
#    n_const
#)
for i in 1:n_opt
    open("../examples/status.txt","a") do f
        println(f,"Combining data...")
    end
    combined_data = FatigueHazards.StepStressRawData(
        combined_stresses,
        combined_cycles
    )
    global full_data = FatigueHazards.partition_time(combined_data)
    full_data.t_norm[2:(end-1)] .= full_data.t_norm[2:(end-1)] .* full_data.t_max ./ initial_data.t_max
    full_data.s_norm[2:(end-1),:] .= full_data.s_norm[2:(end-1),:] .* full_data.s_max ./ initial_data.s_max

    open("../examples/status.txt","a") do f
        println(f,"updating spline grids...")
    end
    FatigueHazards.update_x!(base_haz_spl,full_data.t_norm[1:(end-1)])
    s_unique = sort(unique(full_data.s_norm))
    FatigueHazards.update_x!(risk_spl,s_unique)
    
    s_map = FatigueHazards.map_unique(full_data)
    s_map[2:(end-1),:] .-= 1

    open("../examples/status.txt","a") do f
        println(f,"Finding stepsize...")
    end
    ## initial mcmc samples of beta and gamma
    steps,init_vals,_,_ = FatigueHazards.find_stepsize(
        full_data,
        base_haz_spl,
        risk_spl,
        400,
        70,
        s_map;
        target = [0.44,0.44],
        init_vals=repeat(
            [1.0],
            base_haz_spl.params.num_basis + risk_spl.params.num_basis
        ),
        make_plots=false,
        show_plots=false,
        save_plots=false,
        init=0.001,
        scale = 0.7,
        shape=5.0,
        offset=0.0
    )
    for j in eachindex(steps.beta)
        steps.beta[j] = min(steps.beta[j],1.0)
    end
    for j in eachindex(steps.gamma)
        steps.gamma[j] = min(steps.gamma[j],1.0)
    end

    open("../examples/status.txt","a") do f
        println(f,"Drawing initial samples...")
    end
    samples = FatigueHazards.mcmc_risk_splines(
        full_data,
        base_haz_spl,
        risk_spl,
        20_000,
        steps,
        s_map,
        init_vals
    )

    open("../examples/status.txt","a") do f
        println(f,"Calculating necessary lag...")
    end
    ## calculate necessary lag
    n_burn = 1000
    lag,acf_vals = FatigueHazards.find_lag(
        samples.gamma,
        samples.beta,
        n_burn;
        target=0.1,
        grid_size=2000,
        results=true
    )
    if lag > 2000
        @warn "Large necessary lag detected: $lag"
        println("Reducing to 2000 for computational efficiency")
    end
    lag = min(lag,2000)
    
    n_target = 2500
    max_size = 20_000

    open("../examples/status.txt","a") do f
        println(f,"Drawing bulk i.i.d. samples...")
    end
    bulk_samples = FatigueHazards.bulk_mcmc_risk_splines(
        full_data,
        base_haz_spl,
        risk_spl,
        s_map,
        n_target,
        steps,
        init_vals,
        n_burn,
        lag;
        length_lim=max_size,
        multithread=true
    )

    open("../examples/$(case)beta_samples.txt","a") do f
        for j in 1:n_posterior_save
            println(f,join(string.(bulk_samples.beta[j,:]),','))
        end
    end

    open("../examples/$(case)gamma_samples.txt","a") do f
        for j in 1:n_posterior_save
            println(f,join(string.(bulk_samples.gamma[j,:]),','))
        end
    end

    open("../examples/status.txt","a") do f
        println(f,"Solving optimal design...")
    end
    opt_res = FatigueHazards.optimize_design(
        bulk_samples,
        full_data,
        base_haz_spl,
        risk_spl,
        10;
        s_min=1e3,
        s_max=2e4,
        ds_min=-1e4,
        ds_max=1e4,
        n_max = initial_data.t_max,
        n_const=5000.0,
        n_min=1000,
        n_mcmc=5000,
        n_init=15,
        n_use=3,
        n_rep=3,
        reduce=false,
        multithread=true
    )

    global test_point = opt_res[1]
    gp_inputs = opt_res[2]
    gp_resp = opt_res[3]
    validation_inp = opt_res[4]
    validation_resp = opt_res[5]
    validation_prediction = opt_res[6]
    validation_uncert = opt_res[7]

    open("../examples/$(case)designs.txt","a") do f
        println(f,"$(test_point.s0),$(test_point.ds),$(test_point.n)")
    end

    open("../examples/$(case)gp_inputs.txt","a") do f
        for j in axes(gp_inputs,1)
            println(f,join(string.(gp_inputs[j,:]),','))
        end
    end

    open("../examples/$(case)gp_resp.txt","a") do f
        println(f,join(string.(gp_resp),','))
    end

    open("../examples/$(case)val_inp.txt","a") do f
        for j in axes(validation_inp,2)
            println(f,join(string.(validation_inp[:,j]),','))
        end
    end

    open("../examples/$(case)val_pred.txt","a") do f
        println(f,join(string.(validation_prediction),','))
    end

    open("../examples/$(case)val_resp.txt","a") do f
        println(f,join(string.(validation_resp),','))
    end

    open("../examples/$(case)val_uncert.txt","a") do f
        println(f,join(string.(validation_uncert),','))
    end
    push!(design_points,test_point)

    FatigueHazards.update_x!(risk_spl,general_s_grid)
    FatigueHazards.update_x!(base_haz_spl,general_t_grid)

    open("../examples/status.txt","a") do f
        println(f,"Estimating failure time...")
    end
    for j in eachindex(general_s_grid)
        M_beta = repeat(risk_spl.M[j,:]',length(general_t_grid) - 1)
        for k in 1:n_t_sample
            beta_use = bulk_samples.beta[k,:]
            gamma_use = bulk_samples.gamma[k,:]
            risk_terms = exp.(M_beta * beta_use)

            t,_ = FatigueHazards.sample_t(
                gamma_use,
                base_haz_spl,
                risk_terms,
                general_t_grid,
                1e-6
            )
            t_samples[k,j] = t
        end
    end

    t_samples .*= initial_data.t_max

    open("../examples/$(case)t_samples.txt","a") do f
        for j in axes(t_samples,1)
            println(f,join(string.(t_samples[j,:],",")))
        end
    end
    
    global curr_stresses = copy(full_data.raw.stresses)
    global curr_cycles = copy(full_data.raw.cycles)

    open("../examples/status.txt","a") do f
        println(f,"Simulating data at new design point...")
    end
    global new_data = FatigueHazards.simulate_step_stress(
        damage_rule,
        mat,
        repeat([test_point],n_rep_obs),
        error_dist,
        test_constraints
    ).raw

    combined_stresses = vcat(curr_stresses,new_data.stresses)
    combined_cycles = vcat(curr_cycles,new_data.cycles)
end

@testset "FatigueHazards.jl" begin
    # Write your tests here.
end
