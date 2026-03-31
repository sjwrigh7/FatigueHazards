function eval_entropy(design::StepStressTest,data::StepStressData,n_sim::Int,n_thin::Int,
    n_rep::Int,n_burn::Int,beta_samples::Vector{Float64},gamma_samples::Array{Float64,2},
    spline_design::SplineDesign)

    beta_burn = beta_samples[(n_burn + 1):end]
    gamma_burn = gamma_samples[(n_burn + 1):end,:]

    beta_thin = beta_burn[1:n_thin:end]
    gamma_thin = gamma_burn[1:n_thin:end,:]

    design_norm = StepStressTest(
        design.s0 / data.s_max,
        design.ds / data.s_max,
        design.n / data.t_max
    )

    splines,stress_grid,t_grid = init_design(design_norm,spline_design)
    #println(size(splines.I))

    r_idx = rand(1:length(beta_burn),n_sim)
    r_idx = repeat(r_idx,inner=n_rep)

    r_beta = beta_burn[r_idx]
    r_gamma = gamma_burn[r_idx,:]

    t_samples = Vector{Float64}(undef,n_sim*n_rep)
    ks = Vector{Int}(undef,n_sim*n_rep)

    log_dens = 0.0
    log_marg = 0.0

    ita = 0
    @showprogress "Sampling failure time..." for i in 1:n_sim
        for j in 1:n_rep
            ita += 1
            # pre calculate risk terms over time grid
            risk_terms = exp.(stress_grid[2:end] .* r_beta[i])
            #println(r_gamma[i,:])
            t,k = sample_t(
                r_gamma[i,:],
                splines,
                risk_terms,
                t_grid,
                1e-6
            )
            t_samples[ita] = t
            ks[ita] = k
        end
    end

    combined_time,combined_stress,fail_idx = merge_grids(t_grid,stress_grid,t_samples,ks)

    param_idx = Int.(ceil.(fail_idx ./ n_rep))

    println("Failure idx = ")
    println(fail_idx[1:10])

    println("Param idx = ")
    println(param_idx[1:10])

    update_x!(splines,combined_time)
    ####################
    # try precomputing all vals
    risk_random = [exp.(combined_stress * r_beta[i]) for i in eachindex(r_beta)]
    I_diff_random = [splines.I_diff * r_gamma[i,:] for i in axes(r_gamma,1)]
    M_random = [splines.M * r_gamma[i,:] for i in axes(r_gamma,1)]

    risk_post = [exp.(combined_stress * beta_thin[j]) for j in eachindex(beta_thin)]
    I_diff_post = [splines.I_diff * gamma_thin[j,:] for j in axes(gamma_thin,1)]
    M_post  = [splines.M * gamma_thin[j,:] for j in axes(gamma_thin,1)]
    ####################

    log_dens = Vector{Float64}(undef,n_sim*n_rep)
    log_marg = Array{Float64}(undef,length(beta_thin),n_sim*n_rep)
    mul_blank = Vector{Float64}(undef,maximum(fail_idx))

    #println("risk_random = ",size(risk_random))
    #println("t samples = ",size(t_samples))
    #println("random draws = ",length(r_idx))
    #println("fail idx = ",length(fail_idx))

    ita = 0
    @showprogress "Computing entropy..." for i in 1:n_sim
        for k in 1:n_rep
            ita += 1
            log_dens[ita] = log_density!(
                mul_blank,
                combined_time,
                risk_random[param_idx[ita]],
                I_diff_random[param_idx[ita]],
                M_random[param_idx[ita]],
                fail_idx[ita]
            )
            #log_marg_inner = 0.0
            for j in eachindex(beta_thin)
                log_marg[j,ita] = log_density!(
                    mul_blank,
                    combined_time,
                    risk_post[j],
                    I_diff_post[j],
                    M_post[j],
                    fail_idx[ita]
                )
            end
        end
    end
    #=
    for i in 1:n_sim
        log_dens += log_density(
            t_samples[i],
            combined_stress,
            combined_time,
            r_beta[i],
            r_gamma[i,:],
            splines.I_diff,
            splines.M
        )

        for j in eachindex(beta_thin)
            log_marg += log_density(
                t_samples[i],
                combined_stress,
                combined_time,
                beta_thin[j],
                gamma_thin[j,:],
                splines.I_diff,
                splines.M
            )
        end
    end
    =#

    #entropy = sum(log_marg) / (n_sim * length(beta_thin)) - sum(log_dens) / n_sim

    #objective = entropy / mean(t_samples)
    #println(log_marg)
    #entropy = log_marg / n_sim - log_dens / n_sim
    time_norm = (maximum(t_samples) - minimum(t_samples)) / length(t_samples)
    marg = exp.(log_marg)
    dens = exp.(log_dens)

    marg_norm = vec(mean(marg,dims=1)) / (sum(mean(marg,dims=1)) * time_norm)
    dens_norm = dens / (sum(dens) * time_norm)

    log_marg_norm = log.(marg_norm)
    log_dens_norm = log.(dens_norm)

    ent = mean(log_dens_norm) - mean(log_marg_norm)
    return log_dens_norm,log_marg_norm,combined_time,t_samples
end

function eval_entropy(design::StepStressTest,data::StepStressData,n_sim::Int,lag::Int,
    n_burn::Int,beta_samples::Vector{Float64},gamma_samples::Array{Float64,2},
    spline_design::SplineDesign;n_thin=1)

    beta_burn = beta_samples[(n_burn + 1):end]
    gamma_burn = gamma_samples[(n_burn + 1):end,:]

    #n_samp = length(beta_samples) - n_burn + 1

    #beta_burn = rand(Uniform(1e-4,20),n_samp)
    #gamma_burn = rand(Uniform(1e-4,20),n_samp,size(gamma_samples,2))

    beta_lag = beta_burn[1:lag:end]
    gamma_lag = gamma_burn[1:lag:end,:]
    
    beta_thin = beta_lag[1:n_thin:end]
    gamma_thin = gamma_burn[1:n_thin:end,:]

    design_norm = StepStressTest(
        design.s0 / data.s_max,
        design.ds / data.s_max,
        design.n / data.t_max
    )

    splines,stress_grid,t_grid = init_design(design_norm,spline_design)
    #println(size(splines.I))

    n_sim = min(n_sim,length(beta_lag))
    r_idx = rand(1:length(beta_lag),n_sim)

    r_beta = beta_lag[r_idx]
    r_gamma = gamma_lag[r_idx,:]

    t_samples = Vector{Float64}(undef,n_sim)
    ks = Vector{Int}(undef,n_sim)

    log_dens = 0.0
    log_marg = 0.0

    @showprogress "Sampling failure time..." for i in 1:n_sim
        # pre calculate risk terms over time grid
        risk_terms = exp.(stress_grid[2:end] .* r_beta[i])
        #println(r_gamma[i,:])
        t,k = sample_t(
            r_gamma[i,:],
            splines,
            risk_terms,
            t_grid,
            1e-6
        )
        t_samples[i] = t
        ks[i] = k
    end



    combined_time,combined_stress,merged_idx,new_idx = merge_grids(t_grid,stress_grid,t_samples,ks)

    update_x!(splines,combined_time)
    ####################
    # try precomputing all vals
    risk_random = [exp.(combined_stress * r_beta[i]) for i in eachindex(r_beta)]
    I_diff_random = [splines.I_diff * r_gamma[i,:] for i in axes(r_gamma,1)]
    M_random = [splines.M * r_gamma[i,:] for i in axes(r_gamma,1)]

    risk_post = [exp.(combined_stress * beta_thin[j]) for j in eachindex(beta_thin)]
    I_diff_post = [splines.I_diff * gamma_thin[j,:] for j in axes(gamma_thin,1)]
    M_post  = [splines.M * gamma_thin[j,:] for j in axes(gamma_thin,1)]
    ####################

    log_dens = Vector{Float64}(undef,n_sim)
    log_marg = Array{Float64}(undef,length(beta_thin),n_sim)
    mul_blank = Vector{Float64}(undef,maximum(merged_idx))

    println("risk_random = ",size(risk_random))
    println("combined time = ",size(combined_time))
    println("random draws = ",length(r_idx))
    println("new idx = ",length(new_idx))
    println(maximum(new_idx))
    println("merged idx = ",length(merged_idx))
    println(maximum(merged_idx))
    println("n sim = ",n_sim)
    println("I diff rand = ",size(I_diff_random))

    @showprogress "Computing entropy..." for i in 1:n_sim
        log_dens[i] = log_density!(
            mul_blank,
            combined_time,
            risk_random[new_idx[i]],
            I_diff_random[new_idx[i]],
            M_random[new_idx[i]],
            merged_idx[i]
        )
        #log_marg_inner = 0.0
        for j in eachindex(beta_thin)
            log_marg[j,i] = log_density!(
                mul_blank,
                combined_time,
                risk_post[j],
                I_diff_post[j],
                M_post[j],
                merged_idx[i]
            )
        end
    end

    #entropy = sum(log_marg) / (n_sim * length(beta_thin)) - sum(log_dens) / n_sim

    #objective = entropy / mean(t_samples)
    #println(log_marg)
    #entropy = log_marg / n_sim - log_dens / n_sim
    #time_norm = (maximum(t_samples) - minimum(t_samples)) / length(t_samples)
    #marg = exp.(log_marg)
    #dens = exp.(log_dens)

    #marg_norm = vec(mean(marg,dims=1)) / (sum(mean(marg,dims=1)) * time_norm)
    #dens_norm = dens / (sum(dens) * time_norm)

    #log_marg_norm = log.(marg_norm)
    #log_dens_norm = log.(dens_norm)

    #ent = mean(log_dens_norm) - mean(log_marg_norm)
    return log_dens,log_marg,combined_time,t_samples
end

function get_fact(n,n_theta)
    fact_vals = Array{Int}(undef,n,n_theta)
    for theta in 1:n_theta
        fact_vals[:,theta].= collect(1:n)
    end

    grid = Array{Float64}(undef,n^n_theta,n_theta)

    for j in axes(doe,2)
        grid[:,j] = repeat(fact_fals[:,j],inner=n^(j-1),outer=n^(size(doe,2) - j))
    end

    return grid
end