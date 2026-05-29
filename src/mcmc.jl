"""
    mcmc_linear_risk(...)
CORRECTED implementation of Metropolis-Hastings MCMC sampling of posterior
distributions for survival model parameters.

This method is for the basis coefficients of a set of M & I splines on time domain
and a single coefficient for a linear risk function in stress domain.
"""
function mcmc_linear_risk(data::StepStressData,base_haz_splines::Splines,n_mcmc::Int,steps::StepSize,init_vals)
    
    beta = init_vals[1]
    gamma = init_vals[2:end]
    
    off_gamma = copy(init_vals[2:end])

    gamma_draws = Array{Float64}(undef,n_mcmc,base_haz_splines.params.num_basis)
    beta_draws = Vector{Float64}(undef,n_mcmc)

    gamma_accept = Array{Bool}(undef,size(gamma_draws))
    beta_accept = Vector{Bool}(undef,n_mcmc)
    gamma_accept[1,:] .= true
    beta_accept[1] = true

    beta_draws[1] = beta
    gamma_draws[1,:] .= gamma

    main_time_I_diff_partial = base_haz_splines.I_diff .* gamma'
    off_time_I_diff_partial = base_haz_splines.I_diff .* gamma'

    main_time_I_diff = vec(sum(main_time_I_diff_partial,dims=2))
    off_time_I_diff = vec(sum(off_time_I_diff_partial,dims=2))

    main_time_M_partial = base_haz_splines.M .* gamma'
    off_time_M_partial = base_haz_splines.M .* gamma'

    main_time_M = vec(sum(main_time_M_partial,dims=2))
    off_time_M = vec(sum(off_time_M_partial,dims=2))

    main_inst_risk = Array{Float64}(undef,size(data.s_norm))
    off_inst_risk = similar(main_inst_risk)

    for j in axes(main_inst_risk,2)
        for i in axes(main_inst_risk,1)
            main_inst_risk[i,j] = exp(beta * data.s_norm[i,j])
            off_inst_risk[i,j] = exp(beta * data.s_norm[i,j])
        end
    end

    main_risk_sums = Vector{Float64}(undef,size(main_inst_risk,1))
    off_risk_sums = similar(main_risk_sums)

    for i in 1:(length(main_risk_sums)-1)
        main_risk_sums[i] = sum(main_inst_risk[i,data.in_risk_idx[i]])
        off_risk_sums[i] = sum(off_inst_risk[i,data.in_risk_idx[i]])
    end

    for i in 2:n_mcmc
        for j in 1:base_haz_splines.params.num_basis
            accept = metropolis_gamma!(
                gamma,
                off_gamma,
                main_time_I_diff,
                off_time_I_diff,
                main_time_I_diff_partial,
                off_time_I_diff_partial,
                main_time_M,
                off_time_M,
                main_time_M_partial,
                off_time_M_partial,
                base_haz_splines,
                main_inst_risk,
                main_risk_sums,
                data.fail_idx,
                steps.gamma[j],
                j
            )
            #gamma[j] = gamma_sample
            gamma_draws[i,j] = gamma[j]
            gamma_accept[i,j] = accept
        end

        beta_sample,accept = metropolis_beta!(
            beta,
            main_inst_risk,
            main_risk_sums,
            off_inst_risk,
            off_risk_sums,
            main_time_I_diff,
            main_time_M,
            data.fail_idx,
            data.in_risk_idx,
            data.s_norm,
            steps.beta,
        )

        beta = beta_sample
        beta_draws[i] = beta_sample
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

########################
# refactor more efficient CORRECT mcmc
"""
    mcmc_risk_splines(...)
CORRECTED implementation of Metropolis-Hastings MCMC sampling of
posterior distributions of the survivial model's spline basis coefficients.

This method is for two sets of splines: a combination of M & I splines on time domain
and a set of M splines in stress domain.

"""
function mcmc_risk_splines(
    data::StepStressData,base_haz_splines::Splines,risk_splines::Splines,
    n_mcmc::Int,steps::StepSize,s_map::Array{Int,2},init_vals)
    
    n_risk = risk_splines.params.num_basis
    beta = init_vals[1:n_risk]
    gamma = init_vals[(n_risk + 1):end]

    off_beta = copy(init_vals[1:n_risk])
    off_gamma = copy(init_vals[(n_risk + 1):end])

    gamma_draws = Array{Float64}(undef,n_mcmc,base_haz_splines.params.num_basis)
    beta_draws = Array{Float64}(undef,n_mcmc,n_risk)

    gamma_accept = Array{Bool}(undef,size(gamma_draws))
    beta_accept = Array{Bool}(undef,size(beta_draws))
    gamma_accept[1,:] .= true
    beta_accept[1,:] .= true

    beta_draws[1,:] .= beta
    gamma_draws[1,:] .= gamma

    main_time_I_diff_partial = base_haz_splines.I_diff .* gamma'
    off_time_I_diff_partial = base_haz_splines.I_diff .* gamma'

    main_time_I_diff = vec(sum(main_time_I_diff_partial,dims=2))
    off_time_I_diff = vec(sum(off_time_I_diff_partial,dims=2))

    main_time_M_partial = base_haz_splines.M .* gamma'
    off_time_M_partial = base_haz_splines.M .* gamma'

    main_time_M = vec(sum(main_time_M_partial,dims=2))
    off_time_M = vec(sum(off_time_M_partial,dims=2))

    main_stress_M_partial = risk_splines.M .* beta'
    off_stress_M_partial = risk_splines.M .* beta'

    main_stress_M = vec(sum(main_stress_M_partial,dims=2))
    off_stress_M = vec(sum(off_stress_M_partial,dims=2))

    main_inst_risk = Array{Float64}(undef,size(data.s_norm))
    off_inst_risk = similar(main_inst_risk)

    for j in axes(main_inst_risk,2)
        for i in axes(main_inst_risk,1)
            main_inst_risk[i,j] = exp(main_stress_M[s_map[i,j]])
            off_inst_risk[i,j] = exp(off_stress_M[s_map[i,j]])
        end
    end

    main_risk_sums = Vector{Float64}(undef,size(main_inst_risk,1))
    off_risk_sums = similar(main_risk_sums)

    for i in 1:(length(main_risk_sums)-1)#eachindex(main_risk_sums)
        main_risk_sums[i] = sum(main_inst_risk[i,data.in_risk_idx[i]])
        off_risk_sums[i] = sum(off_inst_risk[i,data.in_risk_idx[i]])
    end

    println(size(main_time_I_diff))
    for i in 2:n_mcmc
        
        for j in 1:base_haz_splines.params.num_basis
            accept = metropolis_gamma!(
                gamma,
                off_gamma,
                main_time_I_diff,
                off_time_I_diff,
                main_time_I_diff_partial,
                off_time_I_diff_partial,
                main_time_M,
                off_time_M,
                main_time_M_partial,
                off_time_M_partial,
                base_haz_splines,
                main_inst_risk,
                main_risk_sums,
                data.fail_idx,
                steps.gamma[j],
                j
            )
            #gamma[j] = gamma_sample
            gamma_draws[i,j] = gamma[j]
            gamma_accept[i,j] = accept
        end

        for j in 1:n_risk
            #println(beta')
            accept = metropolis_beta!(
                beta,
                off_beta,
                main_stress_M,
                off_stress_M,
                main_stress_M_partial,
                off_stress_M_partial,
                main_inst_risk,
                main_risk_sums,
                off_inst_risk,
                off_risk_sums,
                risk_splines,
                main_time_I_diff,
                main_time_M,
                data.fail_idx,
                data.in_risk_idx,
                s_map,
                steps.beta[j],
                j
            )

            #beta[j] = beta_sample
            beta_draws[i,j] = beta[j]
            beta_accept[i,j] = accept
        end
    end
    
    results = PosteriorSamples(
        beta_draws,
        gamma_draws,
        beta_accept,
        gamma_accept
    )

    return results
end

"""
    mcmc_risk_splines!(...)
CORRECTED implementation of Metropolis-Hastings MCMC sampling of
posterior distributions of the survivial model's spline basis coefficients.

This method is for two sets of splines: a combination of M & I splines on time domain
and a set of M splines in stress domain.

"""
function mcmc_risk_splines!(
    beta_draws::Array{Float64,2},gamma_draws::Array{Float64,2},
    beta_accept::Array{Bool,2},gamma_accept::Array{Bool,2},
    main_time_I_diff::Vector{Float64},off_time_I_diff::Vector{Float64},
    main_time_I_diff_partial::Array{Float64,2},off_time_I_diff_partial::Array{Float64,2},
    main_time_M::Vector{Float64},off_time_M::Vector{Float64},
    main_time_M_partial::Array{Float64},off_time_M_partial::Array{Float64,2},
    main_stress_M::Vector{Float64},off_stress_M::Vector{Float64},
    main_stress_M_partial::Array{Float64,2},off_stress_M_partial::Array{Float64,2},
    main_inst_risk::Array{Float64,2},off_inst_risk::Array{Float64,2},
    main_risk_sums::Vector{Float64},off_risk_sums::Vector{Float64},
    data::StepStressData,base_haz_splines::Splines,risk_splines::Splines,
    n_mcmc::Int,steps::StepSize,s_map::Array{Int,2},init_vals)
    
    n_risk = risk_splines.params.num_basis
    beta = init_vals[1:n_risk]
    gamma = init_vals[(n_risk + 1):end]

    off_beta = copy(init_vals[1:n_risk])
    off_gamma = copy(init_vals[(n_risk + 1):end])

    beta_draws[1,:] .= copy(beta)
    gamma_draws[1,:] .= copy(gamma)

    for j in axes(main_inst_risk,2)
        for i in axes(main_inst_risk,1)
            main_inst_risk[i,j] = exp(main_stress_M[s_map[i,j]])
            off_inst_risk[i,j] = exp(off_stress_M[s_map[i,j]])
        end
    end

    for i in 1:(length(main_risk_sums)-1)
        main_risk_sums[i] = sum(main_inst_risk[i,data.in_risk_idx[i]])
        off_risk_sums[i] = sum(off_inst_risk[i,data.in_risk_idx[i]])
    end

    #println(size(main_time_I_diff))
    for i in 2:n_mcmc
        for j in 1:base_haz_splines.params.num_basis
            accept = metropolis_gamma!(
                gamma,
                off_gamma,
                main_time_I_diff,
                off_time_I_diff,
                main_time_I_diff_partial,
                off_time_I_diff_partial,
                main_time_M,
                off_time_M,
                main_time_M_partial,
                off_time_M_partial,
                base_haz_splines,
                main_inst_risk,
                main_risk_sums,
                data.fail_idx,
                steps.gamma[j],
                j
            )

            gamma_draws[i,j] = gamma[j]
            gamma_accept[i,j] = accept
        end

        for j in 1:n_risk
            accept = metropolis_beta!(
                beta,
                off_beta,
                main_stress_M,
                off_stress_M,
                main_stress_M_partial,
                off_stress_M_partial,
                main_inst_risk,
                main_risk_sums,
                off_inst_risk,
                off_risk_sums,
                risk_splines,
                main_time_I_diff,
                main_time_M,
                data.fail_idx,
                data.in_risk_idx,
                s_map,
                steps.beta[j],
                j
            )

            beta_draws[i,j] = beta[j]
            beta_accept[i,j] = accept
        end
    end
    
    results = PosteriorSamples(
        beta_draws,
        gamma_draws,
        beta_accept,
        gamma_accept
    )

    return results
end

"""
    mcmc_linear_risk!(...)
CORRECTED implementation of Metropolis-Hastings MCMC sampling of
posterior distributions of the survivial model's spline basis coefficients.

This method is for two sets of splines: a combination of M & I splines on time domain
and a set of M splines in stress domain.

"""
function mcmc_linear_risk!(
    beta_draws::Vector{Float64},gamma_draws::Array{Float64,2},
    beta_accept::Vector{Bool},gamma_accept::Array{Bool,2},
    main_time_I_diff::Vector{Float64},off_time_I_diff::Vector{Float64},
    main_time_I_diff_partial::Array{Float64,2},off_time_I_diff_partial::Array{Float64,2},
    main_time_M::Vector{Float64},off_time_M::Vector{Float64},
    main_time_M_partial::Array{Float64},off_time_M_partial::Array{Float64,2},
    main_inst_risk::Array{Float64,2},off_inst_risk::Array{Float64,2},
    main_risk_sums::Vector{Float64},off_risk_sums::Vector{Float64},
    data::StepStressData,base_haz_splines::Splines,
    n_mcmc::Int,steps::StepSize,init_vals)
    
    beta = init_vals[1]
    gamma = init_vals[2:end]

    beta_draws[1] = copy(beta)
    gamma_draws[1,:] .= copy(gamma)

    off_gamma = copy(init_vals[2:end])

    for j in axes(main_inst_risk,2)
        for i in axes(main_inst_risk,1)
            main_inst_risk[i,j] = exp(beta * data.s_norm[i,j])
            off_inst_risk[i,j] = exp(beta * data.s_norm[i,j])
        end
    end

    for i in 1:(length(main_risk_sums)-1)
        main_risk_sums[i] = sum(main_inst_risk[i,data.in_risk_idx[i]])
        off_risk_sums[i] = sum(off_inst_risk[i,data.in_risk_idx[i]])
    end

    #println(size(main_time_I_diff))
    for i in 2:n_mcmc
        for j in 1:base_haz_splines.params.num_basis
            accept = metropolis_gamma!(
                gamma,
                off_gamma,
                main_time_I_diff,
                off_time_I_diff,
                main_time_I_diff_partial,
                off_time_I_diff_partial,
                main_time_M,
                off_time_M,
                main_time_M_partial,
                off_time_M_partial,
                base_haz_splines,
                main_inst_risk,
                main_risk_sums,
                data.fail_idx,
                steps.gamma[j],
                j
            )

            gamma_draws[i,j] = gamma[j]
            gamma_accept[i,j] = accept
        end


        beta,accept = metropolis_beta!(
            beta,
            main_inst_risk,
            main_risk_sums,
            off_inst_risk,
            off_risk_sums,
            main_time_I_diff,
            main_time_M,
            data.fail_idx,
            data.in_risk_idx,
            data.s_norm,
            steps.beta,
        )

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
#=
function mcmc_baseline_splines!(beta_draws::Vector{Float64},gamma_draws::Array{Float64,2},
    main_risk::Vector{Float64},off_risk::Vector{Float64},beta_accept::Vector{Bool},
    gamma_accept::Array{Bool,2},data::StepStressData,splines::Splines,
    n_mcmc::Int,steps::StepSize,init_vals)
    fail_indic = sum(data.delta_i[2:(end-1),:],dims=2)

    J = length(data.t_norm)

    beta = init_vals[1]
    gamma = init_vals[2:end]

    @inbounds for j in eachindex(main_risk)
        main_risk[j] = sum_risk(j+1,data.s_norm,beta,data.in_risk_idx)
        off_risk[j] = sum_risk(j+1,data.s_norm,beta,data.in_risk_idx)
    end

    #gamma_draws = Array{Float64}(undef,n_mcmc,splines.params.num_basis)
    #beta_draws = Vector{Float64}(undef,n_mcmc)

    #gamma_accept = Array{Bool}(undef,size(gamma_draws))
    #beta_accept = Array{Bool}(undef,length(beta_draws))
    #gamma_accept[1,:] .- true
    #beta_accept[1] = true

    beta_draws[1] = beta
    gamma_draws[1,:] .= gamma

    @inbounds for i in 2:n_mcmc
        @inbounds for j in 1:splines.params.num_basis
            gamma_sample,acc_sample = metropolis_gamma(
                gamma,
                splines.M,
                splines.I_diff,
                data.s_norm,
                J,
                main_risk,
                fail_indic,
                steps.gamma[j],
                j
            )
            gamma[j] = gamma_sample
            gamma_draws[i,j] = gamma_sample
            gamma_accept[i,j] = acc_sample
        end

        beta_sample,acc_sample = metropolis_beta!(
            main_risk,
            off_risk,
            beta,
            splines.M,
            splines.I_diff,
            data.s_norm,
            fail_indic,
            data.in_risk_idx,
            J,
            gamma,
            steps.beta
        )

        beta = beta_sample
        beta_draws[i] = beta
        beta_accept[i] = acc_sample
    end
end

function mcmc_risk_splines!(beta_draws::Array{Float64,2},gamma_draws::Array{Float64,2},
    main_risk::Vector{Float64},off_risk::Vector{Float64},beta_accept::Array{Bool,2},
    gamma_accept::Array{Bool,2},data::StepStressData,base_haz_splines::Splines,
    risk_splines::Splines,n_mcmc::Int,steps::StepSize,s_map::Array{Int,2},init_vals)
    fail_indic = sum(data.delta_i[2:(end-1),:],dims=2)

    J = length(data.t_norm)
    n_risk = risk_splines.params.num_basis
    beta = init_vals[1:n_risk]
    gamma = init_vals[(n_risk + 1):end]

    @inbounds for j in eachindex(main_risk)
        main_risk[j] = sum_risk(j+1,risk_splines.M,beta,data.in_risk_idx,s_map)
        off_risk[j] = sum_risk(j+1,risk_splines.M,beta,data.in_risk_idx,s_map)
    end

    beta_draws[1,:] .= beta
    gamma_draws[1,:] .= gamma

    @inbounds for i in 2:n_mcmc
        @inbounds for j in 1:base_haz_splines.params.num_basis
            gamma_sample,accept = metropolis_gamma(
                gamma,
                base_haz_splines.M,
                base_haz_splines.I_diff,
                data.s_norm,
                J,
                main_risk,
                fail_indic,
                steps.gamma[j],
                j
            )
            gamma[j] = gamma_sample
            gamma_draws[i,j] = gamma_sample
            gamma_accept[i,j] = accept
        end

        for j in 1:n_risk
            beta_sample,accept = metropolis_beta!(
                main_risk,
                off_risk,
                beta,
                base_haz_splines.M,
                base_haz_splines.I_diff,
                risk_splines.M,
                fail_indic,
                data.in_risk_idx,
                s_map,
                J,
                gamma,
                steps.beta[j],
                j
            )

            beta[j] = beta_sample
            beta_draws[i,j] = beta_sample
            beta_accept[i,j] = accept
        end
    end
end
=#

struct VarsACF
    lags::Vector{Int}
    beta::Union{Array{Float64,1},Array{Float64,2}}
    gamma::Array{Float64,2}
end

function find_lag(gamma,beta,n_burn;target=0.05,grid_size=2000,results=false)
    
    n_tot = size(beta,1) - n_burn

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

    beta_lag_idx = [
        findfirst(x -> x < target,beta_acf[:,i])
        for i in axes(beta_acf,2) ]
    gamma_lag_idx = [
        findfirst(x -> x < target,gamma_acf[:,i])
        for i in axes(gamma_acf,2) ]
    println(beta_lag_idx)

    for i in eachindex(beta_lag_idx)
        if isnothing(beta_lag_idx[i])
            beta_lag_idx[i] = 1
        end
    end
    for i in eachindex(gamma_lag_idx)
        if isnothing(gamma_lag_idx[i])
            gamma_lag_idx[i] = 1
        end
    end

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

function bulk_mcmc_linear_risk(data::StepStressData,base_haz_splines::Splines,
    n_mcmc::Int,steps::StepSize,init_vals,n_burn::Int,lag::Int;
    mem_lim = 0,ele_lim = 0,length_lim = 1_000_000,multithread=true)

    println("Running batch MCMC to draw i.i.d. posteior samples...")
    println("The desired number of i.i.d. samples is ",n_mcmc)

    if (mem_lim != 0) && (ele_lim == 0)
        println("A maximum memory footprint of $(mem_lim / 1e6) MB is specified...")
        max_arr_len = Int(floor(mem_lim / (base_haz_splines.params.num_basis * 8)))
        println("The splines have $(splines.params.num_basis) bases, allowing a maximum batch array length of $(max_arr_len)")
    elseif (mem_lim == 0) && (ele_lim != 0)
        println("A maximum number of $(ele_lim) is specified...")
        max_arr_len = Int(floor(ele_lim / base_haz_splines.params.num_basis))
        println("The splines have $(splines.params.num_basis) bases, allowing a maximum batch array length of $(max_arr_len)")
    elseif (mem_lim != 0) && (ele_lim != 0)
        println("Conflicting limit specifications:")
        println("A maximum memory footprint of $(mem_lim / 1e6) MB is specified")
        println("AND A maximum number of $(ele_lim) is specified...")
        println("Defaulting to memory limit...")
        max_arr_len = Int(floor(mem_lim / (base_haz_splines.params.num_basis * 8)))
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

    
    #if !((n_thread > 0) && (typeof(n_thread) == Int))
    #    @warn "The specified number of threads is not a positive integer..."
    #    println("defaulting to a single thread")
    #    n_thread = 1
    #end
    if multithread
        results = _par_bulk_mcmc_linear_risk(
            data,
            base_haz_splines,
            n_mcmc,
            steps,
            init_vals,
            n_burn,
            lag,
            max_arr_len
        )
    else
        results = _bulk_mcmc_linear_risk(
            data,
            base_haz_splines,
            n_mcmc,
            steps,
            init_vals,
            n_burn,
            lag,
            max_arr_len
        )
    end

    return results

end

function _bulk_mcmc_linear_risk(data::StepStressData,base_haz_splines::Splines,
    n_mcmc::Int,steps::StepSize,init_vals,n_burn::Int,lag::Int,max_arr_len::Int)

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

    beta = init_vals[1]
    gamma = init_vals[2:end]

    gamma_draws = Array{Float64}(undef,n_mcmc,base_haz_splines.params.num_basis)
    beta_draws = Vector{Float64}(undef,n_mcmc)

    gamma_accept = Array{Bool}(undef,size(gamma_draws))
    beta_accept = Vector{Bool}(undef,n_mcmc)
    gamma_accept[1,:] .= true
    beta_accept[1] = true

    beta_draws[1] = beta
    gamma_draws[1,:] .= gamma

    main_time_I_diff_partial = base_haz_splines.I_diff .* gamma'
    off_time_I_diff_partial = base_haz_splines.I_diff .* gamma'

    main_time_I_diff = vec(sum(main_time_I_diff_partial,dims=2))
    off_time_I_diff = vec(sum(off_time_I_diff_partial,dims=2))

    main_time_M_partial = base_haz_splines.M .* gamma'
    off_time_M_partial = base_haz_splines.M .* gamma'

    main_time_M = vec(sum(main_time_M_partial,dims=2))
    off_time_M = vec(sum(off_time_M_partial,dims=2))

    main_inst_risk = Array{Float64}(undef,size(data.s_norm))
    off_inst_risk = similar(main_inst_risk)

    main_risk_sums = Vector{Float64}(undef,size(main_inst_risk,1))
    off_risk_sums = similar(main_risk_sums)

    for i in 1:n_rep
        println("Running batch #$i with an MCMC chain length of $n_run, yielding $n_iid i.i.d. samples")
        start_idx = (i - 1) * n_iid + 1
        stop_idx = i * n_iid
        temp_res = mcmc_linear_risk!(
            beta_draws,
            gamma_draws,
            beta_accept,
            gamma_accept,
            main_time_I_diff,
            off_time_I_diff,
            main_time_I_diff_partial,
            off_time_I_diff_partial,
            main_time_M,
            off_time_M,
            main_time_M_partial,
            off_time_M_partial,
            main_inst_risk,
            off_inst_risk,
            main_risk_sums,
            off_risk_sums,
            data,
            base_haz_splines,
            n_run,
            steps,
            init_vals
        )

        thin_idx = base_range[1:n_iid]
        beta_thin = temp_res.beta[thin_idx]
        gamma_thin = temp_res.gamma[thin_idx,:]
        
        full_beta[start_idx:stop_idx] .= beta_thin
        full_gamma[start_idx:stop_idx,:] .= gamma_thin
    end

    println("Running batch #$(n_rep + 1) with an MCMC chain length of $remainder_sim, yielding $remainder i.i.d. samples")
    remain_res = mcmc_linear_risk!(
        beta_draws,
        gamma_draws,
        beta_accept,
        gamma_accept,
        main_time_I_diff,
        off_time_I_diff,
        main_time_I_diff_partial,
        off_time_I_diff_partial,
        main_time_M,
        off_time_M,
        main_time_M_partial,
        off_time_M_partial,
        main_inst_risk,
        off_inst_risk,
        main_risk_sums,
        off_risk_sums,
        data,
        base_haz_splines,
        n_run,
        steps,
        init_vals
    )

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

function _par_bulk_mcmc_linear_risk(data::StepStressData,base_haz_splines::Splines,
    n_mcmc::Int,steps::StepSize,init_vals,n_burn::Int,lag::Int,max_arr_len::Int)
    
    thread_offset = Threads.nthreads(:interactive)
    n_base = base_haz_splines.params.num_basis

    beta = init_vals[1]
    gamma = init_vals[2:end]

    full_beta = Vector{Float64}(undef,n_mcmc)
    full_gamma = Array{Float64}(undef,n_mcmc,n_base)
    
    beta_iter = [Vector{Float64}(undef,max_arr_len) for _ in 1:Threads.nthreads(:default)]
    gamma_iter = [Array{Float64}(undef,max_arr_len,n_base) for _ in 1:Threads.nthreads(:default)]
    
    beta_accept = [Vector{Bool}(undef,max_arr_len) for _ in 1:Threads.nthreads(:default)]
    gamma_accept = [Array{Bool}(undef,max_arr_len,n_base) for _ in 1:Threads.nthreads(:default)]

    main_time_I_diff_partial_iter = [
        base_haz_splines.I_diff .* gamma' 
        for _ in 1:Threads.nthreads(:default)]
    off_time_I_diff_partial_iter = [
        base_haz_splines.I_diff .* gamma' 
        for _ in 1:Threads.nthreads(:default)]

    main_time_I_diff_iter = [
        vec(sum(main_time_I_diff_partial_iter[i],dims=2)) 
        for i in eachindex(main_time_I_diff_partial_iter)]
    off_time_I_diff_iter = [
        vec(sum(off_time_I_diff_partial_iter[i],dims=2))
        for i in eachindex(off_time_I_diff_partial_iter)]

    main_time_M_partial_iter = [
        base_haz_splines.M .* gamma'
        for _ in 1:Threads.nthreads(:default)]
    off_time_M_partial_iter = [
        base_haz_splines.M .* gamma'
        for _ in 1:Threads.nthreads(:default)]

    main_time_M_iter = [
        vec(sum(main_time_M_partial_iter[i],dims=2))
        for i in eachindex(main_time_M_partial_iter)]
    off_time_M_iter = [
        vec(sum(off_time_M_partial_iter[i],dims=2))
        for i in eachindex(off_time_M_partial_iter)]

    main_inst_risk_iter = [
        Array{Float64}(undef,size(data.s_norm))
        for _ in 1:Threads.nthreads(:default)]
    off_inst_risk_iter = [
        Array{Float64}(undef,size(data.s_norm))
        for _ in 1:Threads.nthreads(:default)]

    main_risk_sums_iter = [
        Vector{Float64}(undef,size(data.s_norm,1))
        for _ in 1:Threads.nthreads(:default)]
    off_risk_sums_iter = [
        Vector{Float64}(undef,size(data.s_norm,1))
        for _ in 1:Threads.nthreads(:default)]

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

    #thread_ids = Vector{Int}(undef,10 * Threads.nthreads())
    #Threads.@threads for i in eachindex(thread_ids)
    #    thread_ids[i] = Threads.threadid()
    #end

    #thread_ids = unique(thread_ids)
    #thread_id_map = Dict([thread_ids[i] => i for i in eachindex(thread_ids)])
    #println(thread_id_map)

    Threads.@threads for i in 1:n_rep
        println("Running batch #$i on thread $(Threads.threadid()) with an MCMC chain length of $n_run, yielding $n_iid i.i.d. samples")
        start_idx = (i - 1) * n_iid + 1
        stop_idx = i * n_iid
        mcmc_linear_risk!(
            beta_iter[Threads.threadid() - thread_offset],
            gamma_iter[Threads.threadid() - thread_offset],
            beta_accept[Threads.threadid() - thread_offset],
            gamma_accept[Threads.threadid() - thread_offset],
            main_time_I_diff_iter[Threads.threadid() - thread_offset],
            off_time_I_diff_iter[Threads.threadid() - thread_offset],
            main_time_I_diff_partial_iter[Threads.threadid() - thread_offset],
            off_time_I_diff_partial_iter[Threads.threadid() - thread_offset],
            main_time_M_iter[Threads.threadid() - thread_offset],
            off_time_M_iter[Threads.threadid() - thread_offset],
            main_time_M_partial_iter[Threads.threadid() - thread_offset],
            off_time_M_partial_iter[Threads.threadid() - thread_offset],
            main_inst_risk_iter[Threads.threadid() - thread_offset],
            off_inst_risk_iter[Threads.threadid() - thread_offset],
            main_risk_sums_iter[Threads.threadid() - thread_offset],
            off_risk_sums_iter[Threads.threadid() - thread_offset],
            data,
            base_haz_splines,
            n_run,
            steps,
            init_vals
        )

        thin_idx = base_range[1:n_iid]
        beta_thin = beta_iter[Threads.threadid() - Threads.nthreads(:interactive)][thin_idx]
        gamma_thin = gamma_iter[Threads.threadid() - Threads.nthreads(:interactive)][thin_idx,:]
        
        full_beta[start_idx:stop_idx] .= beta_thin
        full_gamma[start_idx:stop_idx,:] .= gamma_thin
    end

    println("Running batch #$(n_rep + 1) with an MCMC chain length of $remainder_sim, yielding $remainder i.i.d. samples")
    mcmc_linear_risk!(
        beta_iter[1],
        gamma_iter[1],
        beta_accept[1],
        gamma_accept[1],
        main_time_I_diff_iter[1],
        off_time_I_diff_iter[1],
        main_time_I_diff_partial_iter[1],
        off_time_I_diff_partial_iter[1],
        main_time_M_iter[1],
        off_time_M_iter[1],
        main_time_M_partial_iter[1],
        off_time_M_partial_iter[1],
        main_inst_risk_iter[1],
        off_inst_risk_iter[1],
        main_risk_sums_iter[1],
        off_risk_sums_iter[1],
        data,
        base_haz_splines,
        n_run,
        steps,
        init_vals
    )

    remain_thin_idx = base_range[1:remainder]
    beta_remain = beta_iter[1][remain_thin_idx]
    gamma_remain = gamma_iter[1][remain_thin_idx,:]
    
    remain_start_idx = n_rep * n_iid + 1

    full_beta[remain_start_idx:end] .= beta_remain
    full_gamma[remain_start_idx:end,:] .= gamma_remain

    results = PosteriorIID(
        full_beta,
        full_gamma
    )

    return results
end

function bulk_mcmc_risk_splines(data::StepStressData,base_haz_splines::Splines,
    risk_splines::Splines,s_map::Array{Int,2},n_mcmc::Int,steps::StepSize,init_vals,n_burn::Int,
    lag::Int;mem_lim = 0,ele_lim = 0,length_lim = 1_000_000,multithread=true)

    #println("Running batch MCMC to draw i.i.d. posteior samples...")
    #println("The desired number of i.i.d. samples is ",n_mcmc)

    if (mem_lim != 0) && (ele_lim == 0)
        #println("A maximum memory footprint of $(mem_lim / 1e6) MB is specified...")
        max_arr_len = Int(floor(mem_lim / (splines.params.num_basis * 8)))
        #println("The splines have $(splines.params.num_basis) bases, allowing a maximum batch array length of $(max_arr_len)")
    elseif (mem_lim == 0) && (ele_lim != 0)
        #println("A maximum number of $(ele_lim) is specified...")
        max_arr_len = Int(floor(ele_lim / splines.params.num_basis))
        #println("The splines have $(splines.params.num_basis) bases, allowing a maximum batch array length of $(max_arr_len)")
    elseif (mem_lim != 0) && (ele_lim != 0)
        #println("Conflicting limit specifications:")
        #println("A maximum memory footprint of $(mem_lim / 1e6) MB is specified")
        #println("AND A maximum number of $(ele_lim) is specified...")
        #println("Defaulting to memory limit...")
        max_arr_len = Int(floor(mem_lim / (splines.params.num_basis * 8)))
        #println("The splines have $(splines.params.num_basis) bases, allowing a maximum batch array length of $(max_arr_len)")
    elseif (mem_lim == 0) && (ele_lim == 0) && (length_lim != 0)
        #println("A maximum array length of $length_lim is specified...")
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

    
    #if !((n_thread > 0) && (typeof(n_thread) == Int))
    #    @warn "The specified number of threads is not a positive integer..."
    #    println("defaulting to a single thread")
    #    n_thread = 1
    #end
    if multithread
        results = _par_bulk_mcmc_risk_splines(data,base_haz_splines,risk_splines,s_map,n_mcmc,steps,init_vals,
    n_burn,lag,max_arr_len)
    else
        results = _bulk_mcmc_risk_splines(data,base_haz_splines,risk_splines,s_map,n_mcmc,steps,init_vals,
    n_burn,lag,max_arr_len)
    end

    return results

end
#=
function _bulk_mcmc_risk_splines(data::StepStressData,base_haz_splines::Splines,
    risk_splines::Splines,s_map::Array{Int,2},n_mcmc::Int,steps::StepSize,init_vals,n_burn::Int,
    lag::Int,max_arr_len::Int)

    n_base = base_haz_splines.params.num_basis
    n_risk = risk_splines.params.num_basis

    full_beta = Array{Float64}(undef,n_mcmc,n_risk)
    full_gamma = Array{Float64}(undef,n_mcmc,n_base)
    
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
        temp_res = mcmc_risk_splines(
            data,
            base_haz_splines,
            risk_splines,
            n_run,
            steps,
            s_map,
            init_vals,
        )

        thin_idx = base_range[1:n_iid]
        beta_thin = temp_res.beta[thin_idx,:]
        gamma_thin = temp_res.gamma[thin_idx,:]
        
        full_beta[start_idx:stop_idx,:] .= beta_thin
        full_gamma[start_idx:stop_idx,:] .= gamma_thin
    end

    println("Running batch #$(n_rep + 1) with an MCMC chain length of $remainder_sim, yielding $remainder i.i.d. samples")
    remain_res = mcmc_risk_splines(
        data,
        base_haz_splines,
        risk_splines,
        remainder_sim,
        steps,
        s_map,
        init_vals
    )

    remain_thin_idx = base_range[1:remainder]
    beta_remain = remain_res.beta[remain_thin_idx,:]
    gamma_remain = remain_res.gamma[remain_thin_idx,:]
    
    remain_start_idx = n_rep * n_iid + 1

    full_beta[remain_start_idx:end,:] .= beta_remain
    full_gamma[remain_start_idx:end,:] .= gamma_remain

    results = PosteriorIID(
        full_beta,
        full_gamma
    )

    return results
end

function _par_bulk_mcmc_risk_splines(data::StepStressData,base_haz_splines::Splines,
    risk_splines::Splines,s_map::Array{Int,2},n_mcmc::Int,steps::StepSize,init_vals,n_burn::Int,
    lag::Int,max_arr_len::Int)

    n_base = base_haz_splines.params.num_basis
    n_risk = risk_splines.params.num_basis


    full_beta = Array{Float64}(undef,n_mcmc,n_risk)
    full_gamma = Array{Float64}(undef,n_mcmc,n_base)
    
    beta_iter = [Array{Float64}(undef,max_arr_len,n_risk) for _ in 1:Threads.nthreads(:default)]
    gamma_iter = [Array{Float64}(undef,max_arr_len,n_base) for _ in 1:Threads.nthreads(:default)]
    main_risk_iter = [Vector{Float64}(undef,length(data.t_norm)-2) for _ in 1:Threads.nthreads(:default)]
    off_risk_iter = [Vector{Float64}(undef,length(data.t_norm)-2) for _ in 1:Threads.nthreads(:default)]
    beta_accept = [Array{Bool}(undef,max_arr_len,n_risk) for _ in 1:Threads.nthreads(:default)]
    gamma_accept = [Array{Bool}(undef,max_arr_len,n_base) for _ in 1:Threads.nthreads(:default)]

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

    #thread_ids = Vector{Int}(undef,10 * Threads.nthreads())
    #Threads.@threads for i in eachindex(thread_ids)
    #    thread_ids[i] = Threads.threadid()
    #end

    #thread_ids = unique(thread_ids)
    #thread_id_map = Dict([thread_ids[i] => i for i in eachindex(thread_ids)])
    #println(thread_id_map)

    Threads.@threads for i in 1:n_rep
        println("Running batch #$i on thread $(Threads.threadid()) with an MCMC chain length of $n_run, yielding $n_iid i.i.d. samples")
        start_idx = (i - 1) * n_iid + 1
        stop_idx = i * n_iid
        mcmc_risk_splines!(
            beta_iter[Threads.threadid() - Threads.nthreads(:interactive)],
            gamma_iter[Threads.threadid() - Threads.nthreads(:interactive)],
            main_risk_iter[Threads.threadid() - Threads.nthreads(:interactive)],
            off_risk_iter[Threads.threadid() - Threads.nthreads(:interactive)],
            beta_accept[Threads.threadid() - Threads.nthreads(:interactive)],
            gamma_accept[Threads.threadid() - Threads.nthreads(:interactive)],
            data,
            base_haz_splines,
            risk_splines,
            n_run,
            steps,
            s_map,
            init_vals
        )

        thin_idx = base_range[1:n_iid]
        beta_thin = beta_iter[Threads.threadid() - Threads.nthreads(:interactive)][thin_idx,:]
        gamma_thin = gamma_iter[Threads.threadid() - Threads.nthreads(:interactive)][thin_idx,:]
        
        full_beta[start_idx:stop_idx,:] .= beta_thin
        full_gamma[start_idx:stop_idx,:] .= gamma_thin
    end

    println("Running batch #$(n_rep + 1) with an MCMC chain length of $remainder_sim, yielding $remainder i.i.d. samples")
    mcmc_risk_splines!(
        beta_iter[1],
        gamma_iter[1],
        main_risk_iter[1],
        off_risk_iter[1],
        beta_accept[1],
        gamma_accept[1],
        data,
        base_haz_splines,
        risk_splines,
        remainder_sim,
        steps,
        s_map,
        init_vals
    )

    remain_thin_idx = base_range[1:remainder]
    beta_remain = beta_iter[1][remain_thin_idx,:]
    gamma_remain = gamma_iter[1][remain_thin_idx,:]
    
    remain_start_idx = n_rep * n_iid + 1

    full_beta[remain_start_idx:end,:] .= beta_remain
    full_gamma[remain_start_idx:end,:] .= gamma_remain

    results = PosteriorIID(
        full_beta,
        full_gamma
    )

    return results
end
=#
function _bulk_mcmc_risk_splines(data::StepStressData,base_haz_splines::Splines,
    risk_splines::Splines,s_map::Array{Int,2},n_mcmc::Int,steps::StepSize,init_vals,n_burn::Int,
    lag::Int,max_arr_len::Int)

    n_base = base_haz_splines.params.num_basis
    n_risk = risk_splines.params.num_basis

    beta = init_vals[1:n_risk]
    gamma = init_vals[(n_risk + 1):end]

    full_beta = Array{Float64}(undef,n_mcmc,n_risk)
    full_gamma = Array{Float64}(undef,n_mcmc,n_base)
    
    beta_accept = Array{Bool}(undef,max_arr_len,n_risk)
    gamma_accept = Array{Bool}(undef,max_arr_len,n_base)

    gamma_draws = Array{Float64}(undef,n_mcmc,base_haz_splines.params.num_basis)
    beta_draws = Array{Float64}(undef,n_mcmc,n_risk)

    gamma_accept = Array{Bool}(undef,size(gamma_draws))
    beta_accept = Array{Bool}(undef,size(beta_draws))
    gamma_accept[1,:] .= true
    beta_accept[1,:] .= true

    beta_draws[1,:] .= beta
    gamma_draws[1,:] .= gamma

    main_time_I_diff_partial = base_haz_splines.I_diff .* gamma' 
    off_time_I_diff_partial = base_haz_splines.I_diff .* gamma' 

    main_time_I_diff = vec(sum(main_time_I_diff_partial,dims=2)) 
    off_time_I_diff = vec(sum(off_time_I_diff_partial,dims=2))

    main_time_M_partial = base_haz_splines.M .* gamma'
    off_time_M_partial = base_haz_splines.M .* gamma'

    main_time_M = vec(sum(main_time_M_partial,dims=2))
    off_time_M = vec(sum(off_time_M_partial,dims=2))

    main_stress_M_partial = risk_splines.M .* beta'
    off_stress_M_partial = risk_splines.M .* beta'

    main_stress_M = vec(sum(main_stress_M_partial,dims=2))
    off_stress_M = vec(sum(off_stress_M_partial,dims=2))

    main_inst_risk = Array{Float64}(undef,size(data.s_norm))
    off_inst_risk = Array{Float64}(undef,size(data.s_norm))

    main_risk_sums = Vector{Float64}(undef,size(main_inst_risk,1))
    off_risk_sums = Vector{Float64}(undef,size(main_inst_risk,1))

    n_avail = Int(floor((max_arr_len - n_burn) / lag))
    #println("With a burn value of $n_burn, and a lag of $lag, $n_avail i.i.d. samples can be drawn per batch")
    if n_avail < n_mcmc
        n_iid = n_avail
        n_run = max_arr_len
        n_rep = Int(floor(n_mcmc / n_iid))
        #println("A total of $(n_rep + 1) batches are necessary to achieve the target number of samples")
    else
        n_iid = n_mcmc
        n_run = max_arr_len
        n_rep = 0
        #println("A single batch is sufficient to achieve the target number of samples")
    end

    remainder = n_mcmc - (n_iid * n_rep)
    remainder_sim = remainder * lag + n_burn

    base_range = collect((n_burn + 1):lag:n_run)

    Threads.@threads for i in 1:n_rep
        #println("Running batch #$i / $(n_rep + 1) on thread $(Threads.threadid()) with an MCMC chain length of $n_run, yielding $n_iid i.i.d. samples")
        start_idx = (i - 1) * n_iid + 1
        stop_idx = i * n_iid
        mcmc_risk_splines!(
            beta_draws,
            gamma_draws,
            beta_accept,
            gamma_accept,
            main_time_I_diff,
            off_time_I_diff,
            main_time_I_diff_partial,
            off_time_I_diff_partial,
            main_time_M,
            off_time_M,
            main_time_M_partial,
            off_time_M_partial,
            main_stress_M,
            off_stress_M,
            main_stress_M_partial,
            off_stress_M_partial,
            main_inst_risk,
            off_inst_risk,
            main_risk_sums,
            off_risk_sums,
            data,
            base_haz_splines,
            risk_splines,
            n_run,
            steps,
            s_map,
            init_vals
        )

        thin_idx = base_range[1:n_iid]
        beta_thin = beta_draws[thin_idx,:]
        gamma_thin = gamma_draws[thin_idx,:]
        
        full_beta[start_idx:stop_idx,:] .= beta_thin
        full_gamma[start_idx:stop_idx,:] .= gamma_thin
    end

    #println("Running batch #$(n_rep + 1) (final) with an MCMC chain length of $remainder_sim, yielding $remainder i.i.d. samples")
    mcmc_risk_splines!(
        beta_draws,
        gamma_draws,
        beta_accept,
        gamma_accept,
        main_time_I_diff,
        off_time_I_diff,
        main_time_I_diff_partial,
        off_time_I_diff_partial,
        main_time_M,
        off_time_M,
        main_time_M_partial,
        off_time_M_partial,
        main_stress_M,
        off_stress_M,
        main_stress_M_partial,
        off_stress_M_partial,
        main_inst_risk,
        off_inst_risk,
        main_risk_sums,
        off_risk_sums,
        data,
        base_haz_splines,
        risk_splines,
        n_run,
        steps,
        s_map,
        init_vals
    )

    remain_thin_idx = base_range[1:remainder]
    beta_remain = beta_draws[remain_thin_idx,:]
    gamma_remain = gamma_draws[remain_thin_idx,:]
    
    remain_start_idx = n_rep * n_iid + 1

    full_beta[remain_start_idx:end,:] .= beta_remain
    full_gamma[remain_start_idx:end,:] .= gamma_remain

    results = PosteriorIID(
        full_beta,
        full_gamma
    )

    return results
end

function _par_bulk_mcmc_risk_splines(data::StepStressData,base_haz_splines::Splines,
    risk_splines::Splines,s_map::Array{Int,2},n_mcmc::Int,steps::StepSize,init_vals,n_burn::Int,
    lag::Int,max_arr_len::Int)

    thread_offset = Threads.nthreads(:interactive)
    n_base = base_haz_splines.params.num_basis
    n_risk = risk_splines.params.num_basis

    beta = init_vals[1:n_risk]
    gamma = init_vals[(n_risk + 1):end]
    
    full_beta = Array{Float64}(undef,n_mcmc,n_risk)
    full_gamma = Array{Float64}(undef,n_mcmc,n_base)
    
    beta_iter = [Array{Float64}(undef,max_arr_len,n_risk) for _ in 1:Threads.nthreads(:default)]
    gamma_iter = [Array{Float64}(undef,max_arr_len,n_base) for _ in 1:Threads.nthreads(:default)]
    
    beta_accept = [Array{Bool}(undef,max_arr_len,n_risk) for _ in 1:Threads.nthreads(:default)]
    gamma_accept = [Array{Bool}(undef,max_arr_len,n_base) for _ in 1:Threads.nthreads(:default)]

    #gamma_draws = Array{Float64}(undef,n_mcmc,base_haz_splines.params.num_basis)
    #beta_draws = Array{Float64}(undef,n_mcmc,n_risk)

    #gamma_accept = Array{Bool}(undef,size(gamma_draws[1]))
    #beta_accept = Array{Bool}(undef,size(beta_draws[1]))
    #gamma_accept[1,:] .= true
    #beta_accept[1,:] .= true

    #beta_draws[1,:] .= beta
    #gamma_draws[1,:] .= gamma

    main_time_I_diff_partial_iter = [
        base_haz_splines.I_diff .* gamma' 
        for _ in 1:Threads.nthreads(:default)]
    off_time_I_diff_partial_iter = [
        base_haz_splines.I_diff .* gamma' 
        for _ in 1:Threads.nthreads(:default)]

    main_time_I_diff_iter = [
        vec(sum(main_time_I_diff_partial_iter[i],dims=2)) 
        for i in eachindex(main_time_I_diff_partial_iter)]
    off_time_I_diff_iter = [
        vec(sum(off_time_I_diff_partial_iter[i],dims=2))
        for i in eachindex(off_time_I_diff_partial_iter)]

    main_time_M_partial_iter = [
        base_haz_splines.M .* gamma'
        for _ in 1:Threads.nthreads(:default)]
    off_time_M_partial_iter = [
        base_haz_splines.M .* gamma'
        for _ in 1:Threads.nthreads(:default)]

    main_time_M_iter = [
        vec(sum(main_time_M_partial_iter[i],dims=2))
        for i in eachindex(main_time_M_partial_iter)]
    off_time_M_iter = [
        vec(sum(off_time_M_partial_iter[i],dims=2))
        for i in eachindex(off_time_M_partial_iter)]

    main_stress_M_partial_iter = [
        risk_splines.M .* beta'
        for _ in 1:Threads.nthreads(:default)]
    off_stress_M_partial_iter = [
        risk_splines.M .* beta'
        for _ in 1:Threads.nthreads(:default)]

    main_stress_M_iter = [
        vec(sum(main_stress_M_partial_iter[i],dims=2))
        for i in eachindex(main_stress_M_partial_iter)]
    off_stress_M_iter = [
        vec(sum(off_stress_M_partial_iter[i],dims=2))
        for i in eachindex(off_stress_M_partial_iter)]

    main_inst_risk_iter = [
        Array{Float64}(undef,size(data.s_norm))
        for _ in 1:Threads.nthreads(:default)]
    off_inst_risk_iter = [
        Array{Float64}(undef,size(data.s_norm))
        for _ in 1:Threads.nthreads(:default)]

    main_risk_sums_iter = [
        Vector{Float64}(undef,size(data.s_norm,1))
        for _ in 1:Threads.nthreads(:default)]
    off_risk_sums_iter = [
        Vector{Float64}(undef,size(data.s_norm,1))
        for _ in 1:Threads.nthreads(:default)]


    n_avail = Int(floor((max_arr_len - n_burn) / lag))
    #println("With a burn value of $n_burn, and a lag of $lag, $n_avail i.i.d. samples can be drawn per batch")
    if n_avail < n_mcmc
        n_iid = n_avail
        n_run = max_arr_len
        n_rep = Int(floor(n_mcmc / n_iid))
        #println("A total of $(n_rep + 1) batches are necessary to achieve the target number of samples")
    else
        n_iid = n_mcmc
        n_run = max_arr_len
        n_rep = 0
        #println("A single batch is sufficient to achieve the target number of samples")
    end

    remainder = n_mcmc - (n_iid * n_rep)
    remainder_sim = remainder * lag + n_burn

    base_range = collect((n_burn + 1):lag:n_run)

    Threads.@threads for i in 1:n_rep
        #println("Running batch #$i / $(n_rep + 1) on thread $(Threads.threadid()) with an MCMC chain length of $n_run, yielding $n_iid i.i.d. samples")
        start_idx = (i - 1) * n_iid + 1
        stop_idx = i * n_iid
        mcmc_risk_splines!(
            beta_iter[Threads.threadid() - thread_offset],
            gamma_iter[Threads.threadid() - thread_offset],
            beta_accept[Threads.threadid() - thread_offset],
            gamma_accept[Threads.threadid() - thread_offset],
            main_time_I_diff_iter[Threads.threadid() - thread_offset],
            off_time_I_diff_iter[Threads.threadid() - thread_offset],
            main_time_I_diff_partial_iter[Threads.threadid() - thread_offset],
            off_time_I_diff_partial_iter[Threads.threadid() - thread_offset],
            main_time_M_iter[Threads.threadid() - thread_offset],
            off_time_M_iter[Threads.threadid() - thread_offset],
            main_time_M_partial_iter[Threads.threadid() - thread_offset],
            off_time_M_partial_iter[Threads.threadid() - thread_offset],
            main_stress_M_iter[Threads.threadid() - thread_offset],
            off_stress_M_iter[Threads.threadid() - thread_offset],
            main_stress_M_partial_iter[Threads.threadid() - thread_offset],
            off_stress_M_partial_iter[Threads.threadid() - thread_offset],
            main_inst_risk_iter[Threads.threadid() - thread_offset],
            off_inst_risk_iter[Threads.threadid() - thread_offset],
            main_risk_sums_iter[Threads.threadid() - thread_offset],
            off_risk_sums_iter[Threads.threadid() - thread_offset],
            data,
            base_haz_splines,
            risk_splines,
            n_run,
            steps,
            s_map,
            init_vals
        )

        thin_idx = base_range[1:n_iid]
        beta_thin = beta_iter[Threads.threadid() - thread_offset][thin_idx,:]
        gamma_thin = gamma_iter[Threads.threadid() - thread_offset][thin_idx,:]
        
        full_beta[start_idx:stop_idx,:] .= beta_thin
        full_gamma[start_idx:stop_idx,:] .= gamma_thin
    end

    #println("Running batch #$(n_rep + 1) (final) with an MCMC chain length of $remainder_sim, yielding $remainder i.i.d. samples")
    mcmc_risk_splines!(
        beta_iter[1],
        gamma_iter[1],
        beta_accept[1],
        gamma_accept[1],
        main_time_I_diff_iter[1],
        off_time_I_diff_iter[1],
        main_time_I_diff_partial_iter[1],
        off_time_I_diff_partial_iter[1],
        main_time_M_iter[1],
        off_time_M_iter[1],
        main_time_M_partial_iter[1],
        off_time_M_partial_iter[1],
        main_stress_M_iter[1],
        off_stress_M_iter[1],
        main_stress_M_partial_iter[1],
        off_stress_M_partial_iter[1],
        main_inst_risk_iter[1],
        off_inst_risk_iter[1],
        main_risk_sums_iter[1],
        off_risk_sums_iter[1],
        data,
        base_haz_splines,
        risk_splines,
        n_run,
        steps,
        s_map,
        init_vals
    )

    remain_thin_idx = base_range[1:remainder]
    beta_remain = beta_iter[1][remain_thin_idx,:]
    gamma_remain = gamma_iter[1][remain_thin_idx,:]
    
    remain_start_idx = n_rep * n_iid + 1

    full_beta[remain_start_idx:end,:] .= beta_remain
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

#################
# adaptive stepsize MCMC
"""
    mcmc_risk_splines(...)
CORRECTED implementation of Metropolis-Hastings MCMC sampling of
posterior distributions of the survivial model's spline basis coefficients.

This method is for two sets of splines: a combination of M & I splines on time domain
and a set of M splines in stress domain.

"""
function adaptive_mcmc_risk_splines(
    data::StepStressData,base_haz_splines::Splines,risk_splines::Splines,
    n_mcmc::Int,s_map::Array{Int,2},init_vals;n_init=300,target=0.234)

    n_risk = risk_splines.params.num_basis
    beta = init_vals[1:n_risk]
    gamma = init_vals[(n_risk + 1):end]

    steps = StepSize(
        1e-3 * copy(beta),
        1e-3 * copy(gamma)
    )

    off_beta = copy(init_vals[1:n_risk])
    off_gamma = copy(init_vals[(n_risk + 1):end])

    gamma_draws = Array{Float64}(undef,n_mcmc,base_haz_splines.params.num_basis)
    beta_draws = Array{Float64}(undef,n_mcmc,n_risk)

    gamma_accept = Array{Bool}(undef,size(gamma_draws))
    beta_accept = Array{Bool}(undef,size(beta_draws))
    gamma_accept[1,:] .= true
    beta_accept[1,:] .= true

    beta_draws[1,:] .= beta
    gamma_draws[1,:] .= gamma

    main_time_I_diff_partial = base_haz_splines.I_diff .* gamma'
    off_time_I_diff_partial = base_haz_splines.I_diff .* gamma'

    main_time_I_diff = vec(sum(main_time_I_diff_partial,dims=2))
    off_time_I_diff = vec(sum(off_time_I_diff_partial,dims=2))

    main_time_M_partial = base_haz_splines.M .* gamma'
    off_time_M_partial = base_haz_splines.M .* gamma'

    main_time_M = vec(sum(main_time_M_partial,dims=2))
    off_time_M = vec(sum(off_time_M_partial,dims=2))

    main_stress_M_partial = risk_splines.M .* beta'
    off_stress_M_partial = risk_splines.M .* beta'

    main_stress_M = vec(sum(main_stress_M_partial,dims=2))
    off_stress_M = vec(sum(off_stress_M_partial,dims=2))

    main_inst_risk = Array{Float64}(undef,size(data.s_norm))
    off_inst_risk = similar(main_inst_risk)

    for j in axes(main_inst_risk,2)
        for i in axes(main_inst_risk,1)
            main_inst_risk[i,j] = exp(main_stress_M[s_map[i,j]])
            off_inst_risk[i,j] = exp(off_stress_M[s_map[i,j]])
        end
    end

    main_risk_sums = Vector{Float64}(undef,size(main_inst_risk,1))
    off_risk_sums = similar(main_risk_sums)

    for i in 1:(length(main_risk_sums)-1)#eachindex(main_risk_sums)
        main_risk_sums[i] = sum(main_inst_risk[i,data.in_risk_idx[i]])
        off_risk_sums[i] = sum(off_inst_risk[i,data.in_risk_idx[i]])
    end
    
    for i in 2:n_init
        for j in 1:base_haz_splines.params.num_basis
            accept = metropolis_gamma!(
                gamma,
                off_gamma,
                main_time_I_diff,
                off_time_I_diff,
                main_time_I_diff_partial,
                off_time_I_diff_partial,
                main_time_M,
                off_time_M,
                main_time_M_partial,
                off_time_M_partial,
                base_haz_splines,
                main_inst_risk,
                main_risk_sums,
                data.fail_idx,
                steps.gamma[j],
                j
            )
            #gamma[j] = gamma_sample
            gamma_draws[i,j] = gamma[j]
            gamma_accept[i,j] = accept
        end

        for j in 1:n_risk
            accept = metropolis_beta!(
                beta,
                off_beta,
                main_stress_M,
                off_stress_M,
                main_stress_M_partial,
                off_stress_M_partial,
                main_inst_risk,
                main_risk_sums,
                off_inst_risk,
                off_risk_sums,
                risk_splines,
                main_time_I_diff,
                main_time_M,
                data.fail_idx,
                data.in_risk_idx,
                s_map,
                steps.beta[j],
                j
            )

            #beta[j] = beta_sample
            beta_draws[i,j] = beta[j]
            beta_accept[i,j] = accept
        end
    end
    
    mean_tracker = Array{Float64}(undef,n_mcmc,size(beta_draws,2)+size(gamma_draws,2))
    var_tracker = similar(mean_tracker)

    mean_betas = vec(mean(beta_draws[1:n_init,:],dims=1))
    var_betas = vec(var(beta_draws[1:n_init,:],dims=1))

    mean_gammas = vec(mean(gamma_draws[1:n_init,:],dims=1))
    var_gammas = vec(var(gamma_draws[1:n_init,:],dims=1))

    mean_tracker[1:n_init,:] .= hcat(mean_betas',mean_gammas')
    var_tracker[1:n_init,:] .= hcat(var_betas',var_gammas')

    learning_rate = 0.01

    for i in (n_init + 1):n_mcmc
        for j in 1:base_haz_splines.params.num_basis
            steps.gamma[j] = sqrt(var_gammas[j])

            accept = metropolis_gamma!(
                gamma,
                off_gamma,
                main_time_I_diff,
                off_time_I_diff,
                main_time_I_diff_partial,
                off_time_I_diff_partial,
                main_time_M,
                off_time_M,
                main_time_M_partial,
                off_time_M_partial,
                base_haz_splines,
                main_inst_risk,
                main_risk_sums,
                data.fail_idx,
                steps.gamma[j],
                j
            )
            #gamma[j] = gamma_sample
            gamma_draws[i,j] = gamma[j]
            gamma_accept[i,j] = accept

            var_gammas[j] = var_gammas[j] + learning_rate * ((gamma[j] - mean_gammas[j])^2 - var_gammas[j])
            mean_gammas[j] = mean_gammas[j] + learning_rate * (gamma[j] - mean_gammas[j])

            mean_tracker[i,j + n_risk] = mean_gammas[j]
            var_tracker[i,j + n_risk] = var_gammas[j]
        end

        for j in 1:n_risk
            steps.beta[j] = sqrt(var_betas[j])

            accept = metropolis_beta!(
                beta,
                off_beta,
                main_stress_M,
                off_stress_M,
                main_stress_M_partial,
                off_stress_M_partial,
                main_inst_risk,
                main_risk_sums,
                off_inst_risk,
                off_risk_sums,
                risk_splines,
                main_time_I_diff,
                main_time_M,
                data.fail_idx,
                data.in_risk_idx,
                s_map,
                steps.beta[j],
                j
            )

            #beta[j] = beta_sample
            beta_draws[i,j] = beta[j]
            beta_accept[i,j] = accept

            var_betas[j] = var_betas[j] + learning_rate * ((beta[j] - mean_betas[j])^2 - var_gammas[j])
            mean_betas[j] = mean_betas[j] + learning_rate * (beta[j] - mean_betas[j])

            mean_tracker[i,j] = mean_betas[j]
            var_tracker[i,j] = var_betas[j]
        end
    end
    
    results = PosteriorSamples(
        beta_draws,
        gamma_draws,
        beta_accept,
        gamma_accept
    )

    return results,mean_tracker,var_tracker
end