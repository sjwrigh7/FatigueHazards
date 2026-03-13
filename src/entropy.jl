function eval_entropy(design::StepStressTest,data::StepStressData,n_sim::Int,n_thin::Int,
    n_burn::Int,beta_samples::Vector{Float64},gamma_samples::Vector{Float64},
    spline_design::SplineDesign)

    beta_burn = beta_samples[(n_burn + 1):end]
    gamma_burn = beta_samples[(n_burn + 1):end,:]

    beta_thin = beta_burn[1:n_thin:end]
    gamma_thin = gamma_burn[1:n_thin:end,:]

    design_norm = StepStressTest(
        design.s0 / data.s_max,
        design.ds / data.s_max,
        design.n / data.t_max
    )

    splines,stress_grid,t_grid = init_design(design_norm,spline_design)

    r_idx = rand(1:length(beta_burn),n_sim)

    r_beta = beta_burn[r_idx]
    r_gamma = gamma_burn[r_idx,:]

    t_samples = Vector{Float64}(undef,n_sim)
    ks = Vector{Int}(undef,n_sim)

    log_dens = 0.0
    log_marg = 0.0

    for i in 1:n_sim

        # pre calculate risk terms over time grid
        risk_terms = exp.(stress_grid[2:(end-1)] .* r_beta[i])

        t,k = sample_t(
            r_gamma[i,:],
            splines,
            risk_terms,
            t_grid,
            tol=1e-6
        )
        t_samples[i] = t
        ts[i] = k
    end



    combined_time,combined_stress = merge_grids(time_grid,stress_grid,t_samples,ks)

    update_x!(splines,combined_time)

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

    entropy = sum(log_marg) / (n_sim * lengtht(beta_thin)) - sum(log_dens) / n_sim

    objective = entropy / mean(y_samples)

    return objective
end