struct PosteriorSamples
    beta::Union{Array{Float64,1},Array{Float64,2}}
    gamma::Array{Float64,2}
    beta_accept::Union{Array{Bool,1},Array{Bool,2}}
    gamma_accept::Array{Bool,2}
end

struct PosteriorIID
    beta::Union{Array{Float64,1},Array{Float64,2}}
    gamma::Array{Float64,2}
end

function mcmc_baseline_splines(data,splines,n_mcmc,steps,init_vals)
    fail_indic = sum(data.delta_i[2:(end-1),:],dims=2)

    J = length(data.t_norm)

    beta = init_vals[1]
    gamma = init_vals[2:end]

    risk_terms = [sum_risk(j,data.s_norm,beta,data.delta_i) for j in 2:(J-1)]

    gamma_draws = Array{Float64}(undef,n_mcmc,splines.params.num_basis)
    beta_draws = Vector{Float64}(undef,n_mcmc)

    gamma_accept = Array{Bool}(undef,size(gamma_draws))
    beta_accept = Array{Bool}(undef,length(beta_draws))
    gamma_accept[1,:] .- true
    beta_accept[1] = true

    beta_draws[1] = beta
    gamma_draws[1,:] .= gamma

    @showprogress "MCMC Iterating..." for i in 2:n_mcmc
        for j in 1:splines.params.num_basis
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
            data.delta_i,
            J,
            gamma,
            steps[end]
        )

        beta = beta_sample
        beta_draws[i] = beta
        beta_accept[i] = accept
    end
    
    results = PosteriorSamples(
        beta_draws,
        gamma_draws,
        beta_accept,
        gamma_accept
    )

    return results
end

struct VarsACF
    lags::Vector{Int}
    beta::Union{Array{Float64,1},Array{Float64,2}}
    gamma::Array{Float64,2}
end

function find_lag(gamma,beta,n_burn;target=0.05,grid_size=2000,results=false)
    
    n_tot = length(beta) - n_burn

    max_lag = log(10,n_tot / 2)

    lag_vals = sort(
        unique(
            round.(
                Int,
                10 .^ collect(
                    range(
                        start=0,
                        stop=max_lag,
                        length=grid_size
                    )
                )
            )
        )
    )

    beta_acf = autocor(beta,lag_vals,demean=true)
    gamma_acf = autocor(gamma,lag_vals,demean=true)

    beta_lag_idx = findfirst(x -> x < target,beta_acf)
    gamma_lag_idx = [
        findfirst(x -> x < target,gamma_acf[:,i])
        for i in axes(gamma_acf,2) ]

    beta_lag = lag_vals[beta_lag_idx]
    gamma_lag = lag_vals[gamma_lag_idx]

    lag_use = maximum(
        vcat(
            beta_lag,
            gamma_lag
        )
    )
    acf_vals = VarsACF(
        lag_vals,
        beta_acf,
        gamma_acf
    )
    if results
        return lag_use,acf_vals
    else
        return lag_use
    end
end

function bulk_mcmc_baseline_splines(data,splines,n_mcmc,steps,init_vals,
    n_burn,lag;mem_lim = 0,ele_lim = 0,length_lim = 1_000_000)

    println("Running batch MCMC to draw i.i.d. posteior samples...")
    println("The desired number of i.i.d. samples is ",n_mcmc)

    if (mem_lim != 0) && (ele_lim == 0)
        println("A maximum memory footprint of $(mem_lim / 1e6) MB is specified...")
        max_arr_len = Int(floor(mem_lim / (splines.params.num_basis * 8)))
        println("The splines have $(splines.params.num_basis) bases, allowing a maximum batch array length of $(max_arr_len)")
    elseif (mem_lim == 0) && (ele_lim != 0)
        println("A maximum number of $(ele_lim) is specified...")
        max_arr_len = Int(floor(ele_lim / splines.params.num_basis))
        println("The splines have $(splines.params.num_basis) bases, allowing a maximum batch array length of $(max_arr_len)")
    elseif (mem_lim != 0) && (ele_lim != 0)
        println("Conflicting limit specifications:")
        println("A maximum memory footprint of $(mem_lim / 1e6) MB is specified")
        println("AND A maximum number of $(ele_lim) is specified...")
        println("Defaulting to memory limit...")
        max_arr_len = Int(floor(mem_lim / (splines.params.num_basis * 8)))
        println("The splines have $(splines.params.num_basis) bases, allowing a maximum batch array length of $(max_arr_len)")
    elseif (mem_lim == 0) && (ele_lim == 0) && (length_lim != 0)
        println("A maximum array length of $length_lim is specified...")
        max_arr_len = length_lim
    elseif (mem_lim == 0) && (ele_lim == 0) && (length_lim == 0)
        @warn "No limit specified for MCMC...
        Defaulting to a maximum array length of 10000000 for safety"
        max_arr_len = 10_000_000
    else
        @warn "Unexpected specification of MCMC batch limits...
        Defaulting to a maximum array length of 1000000"
        max_arr_len = 1_000_000
    end

    full_beta = Vector{Float64}(undef,n_mcmc)
    full_gamma = Array{Float64}(undef,n_mcmc,splines.params.num_basis)
    
    n_avail = Int(floor((max_arr_len - n_burn) / lag))
    println("With a burn value of $n_burn, and a lag of $lag, $n_avail i.i.d. samples can be drawn per batch")
    if n_avail < n_mcmc
        n_iid = n_avail
        n_run = max_arr_len
        n_rep = Int(floor(n_mcmc / n_iid))
        println("A total of $(n_rep + 1) batches are necessary to achieve the target number of samples")
    else
        n_iid = n_mcmc
        n_run = max_arr_len
        n_rep = 0
        println("A single batch is sufficient to achieve the target number of samples")
    end

    remainder = n_mcmc - (n_iid * n_rep)
    remainder_sim = remainder * lag + n_burn

    base_range = collect((n_burn + 1):lag:n_run)

    for i in 1:n_rep
        println("Running batch #$i with an MCMC chain length of $n_run, yielding $n_iid i.i.d. samples")
        start_idx = (i - 1) * n_iid + 1
        stop_idx = i * n_iid
        temp_res = mcmc_baseline_splines(data,splines,n_run,steps,init_vals)

        thin_idx = base_range[1:n_iid]
        beta_thin = temp_res.beta[thin_idx]
        gamma_thin = temp_res.gamma[thin_idx,:]
        
        full_beta[start_idx:stop_idx] .= beta_thin
        full_gamma[start_idx:stop_idx,:] .= gamma_thin
    end

    println("Running batch #$(n_rep + 1) with an MCMC chain length of $remainder_sim, yielding $remainder i.i.d. samples")
    remain_res = mcmc_baseline_splines(data,splines,remainder_sim,steps,init_vals)

    remain_thin_idx = base_range[1:remainder]
    beta_remain = remain_res.beta[remain_thin_idx]
    gamma_remain = remain_res.gamma[remain_thin_idx,:]
    
    remain_start_idx = n_rep * n_iid + 1

    full_beta[remain_start_idx:end] .= beta_remain
    full_gamma[remain_start_idx:end,:] .= gamma_remain

    results = PosteriorIID(
        full_beta,
        full_gamma
    )

    return results
end

function geweke_statistic(samples::Vector{Float64};burn = 0,norm=false)
    n = length(samples)
    if burn == 0
        burn = round(Int,0.5 * n)
    else
        if (n - burn) < 70
            @warn "less than 35 samples are avaiable for each subset of the
            Monte Carlo samples, which may violate the normality assumption.
            Consider specifying a reduced burn number of use more samples"
        end
    end
    n_set = round(
        Int,
        floor(0.5 * (n - burn))
    )
    
    set1_idx = (burn + 1):(burn + n_set)
    set2_idx = (burn + n_set + 1):(burn + 2 * n_set)

    set1_mean = mean(samples[set1_idx])
    set2_mean = mean(samples[set2_idx])

    set1_var = var(samples[set1_idx])
    set2_var = var(samples[set2_idx])

    if norm
        z = (set1_mean - set2_mean) / 
            sqrt(
                set1_var + set2_var
            )
    else
        z = (set1_mean - set2_mean) / 
            sqrt(
                set1_var / n_set + 
                set2_var / n_set
            )
    end
    
    if z > 1.645 || z < -1.645
        println("Z score lies outside of 95% CI, indicating a lack of convergence")
        println("Z = $z")
        return false
    else
        println("Z score lies within 95% CI, indicating convergence")
        println("Z = $z")
        return true
    end
end