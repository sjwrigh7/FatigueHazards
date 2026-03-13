function mcmc_baseline_splines(data,splines,n_mcmc,steps,init_vals)
    fail_indic = sum(delta_i[2:(end-1),:],dims=2)

    J = length(T)

    beta = init_vals[1]
    gamma = init_vals[2:end]

    risk_terms = [calc_Aj(j,data.s_norm,beta,data.delta_i) for j in 2:(J-1)]

    gamma_draws = Array{Float64}(undef,n_mcmc,splines.params.num_basis)
    beta_draws = Vector{Float64}(undef,n_mcmc)

    gamma_accept = Array{Bool}(undef,size(gamma_draws))
    beta_accept = Array{Float64}(undef,length(beta_draws))
    gamma_accept[1,:] .- true
    beta_accept[1] = true

    beta_draws[1] = beta
    gamma_draws[1,:] .= gamma

    @showprogress "MCMC Iterating..." for i in 2:n_mcmc
        for j in 1:n
            gamma_sample,accept = metropolis_gamma(
                gamma,
                splines.M,
                splines.I_diff,
                data.s_norm,
                J,
                risk_terms,
                fail_indic,
                steps[j],
                j
            )
            gamma[j] = gamma_sample
            gamma_draws[i,j] = gamma_sample
            gamma_accept[i,j] = accept
        end

        beta_sample,accept,risk_terms = metropolis_beta(
            beta,
            splines.M,
            splines.I_diff,
            data.s_norm,
            fail_indic,
            J,
            gamma,
            steps[end]
        )

        beta = beta_sample
        beta_draws[i] = beta
        beta_accept[i] = accept
    end
    
    return gamma_draws,beta_draws,gamma_accept,beta_accept,M_star,I_star,knot_grid
end