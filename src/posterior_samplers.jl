"""
    metropolis_gamma(...)
CORRECTED implementation of a Metropolis-Hastings update for a single time domain
spline basis coefficient.

This method is for a survival model using an M-spline risk function
"""
function metropolis_gamma(gamma::Vector{Float64},M::Array{Float64,2},I_diff::Array{Float64,2},
    M_beta::Array{Float64,2},beta::Vector{Float64},fail_idx::Vector{Int},s_map::Array{Int,2},
    step::Float64,j::Int,priors::Priors)
    
    current_gamma = gamma
    proposed_gamma = copy(current_gamma)

    current_transformed = log(gamma[j])
    proposed_transformed = rand(Normal(current_transformed,step))
    new_gamma = exp(proposed_transformed)
    proposed_gamma[j] = new_gamma

    current_lik = log_lik(current_gamma,M,I_diff,M_beta,beta,fail_idx,s_map)
    proposed_lik = log_lik(proposed_gamma,M,I_diff,M_beta,beta,fail_idx,s_map)

    log_jump_current = logpdf(
        Normal(
            current_transformed,
            step
        ),
        proposed_transformed
    ) - log(proposed_gamma[j])
    
    log_jump_proposed = logpdf(
        Normal(
            proposed_transformed,
            step
        ),
        current_transformed
    ) - log(current_gamma[j])
    
    log_lik_ratio = proposed_lik - current_lik
    log_jump_ratio = log_jump_proposed - log_jump_current
    log_prior_ratio = logpdf(priors.gamma[idx],proposed_gamma[idx]) -
        logpdf(priors.gamma[idx],current_gamma[idx])
    acceptance_ratio = log_lik_ratio + log_jump_ratio + log_prior_ratio

    log_u = log(rand(Uniform(0.0,1.0)))
    accept = log_u <= acceptance_ratio

    if accept
        return_gamma = proposed_gamma[j]
    else
        return_gamma = current_gamma[j]
    end

    return return_gamma,accept
end

"""
    metropolis_gamma(...)
CORRECTED implementation of a Metropolis-Hastings update for a single time domain
spline basis coefficient.

This method is for a survival model using a linear risk function
"""
function metropolis_gamma(gamma::Vector{Float64},M::Array{Float64,2},I_diff::Array{Float64,2},
    stresses::Array{Float64,2},beta::Float64,fail_idx::Vector{Int},
    step::Float64,j::Int,priors::Priors)
    
    current_gamma = gamma
    proposed_gamma = copy(current_gamma)

    current_transformed = log(gamma[j])
    proposed_transformed = rand(Normal(current_transformed,step))
    new_gamma = exp(proposed_transformed)
    proposed_gamma[j] = new_gamma

    current_lik = log_lik(current_gamma,M,I_diff,stresses,beta,fail_idx)
    proposed_lik = log_lik(proposed_gamma,M,I_diff,stresses,beta,fail_idx)

    log_jump_current = logpdf(
        Normal(
            current_transformed,
            step
        ),
        proposed_transformed
    ) - log(proposed_gamma[j])
    
    log_jump_proposed = logpdf(
        Normal(
            proposed_transformed,
            step
        ),
        current_transformed
    ) - log(current_gamma[j])
    
    log_lik_ratio = proposed_lik - current_lik
    log_jump_ratio = log_jump_proposed - log_jump_current
    log_prior_ratio = logpdf(priors.gamma[idx],proposed_gamma[idx]) -
        logpdf(priors.gamma[idx],current_gamma[idx])
    acceptance_ratio = log_lik_ratio + log_jump_ratio + log_prior_ratio

    log_u = log(rand(Uniform(0.0,1.0)))
    accept = log_u <= acceptance_ratio

    if accept
        return_gamma = proposed_gamma[j]
    else
        return_gamma = current_gamma[j]
    end

    return return_gamma,accept
end
"""
    metropolis_beta(...)
CORRECTED implementation for a Metropolis-Hastings update of a single
stress domain basis coefficient.

This method is for a survival model using an M-spline risk function.
"""
function metropolis_beta(beta::Vector{Float64},M::Array{Float64,2},I_diff::Array{Float64,2},
    M_beta::Array{Float64,2},fail_idx::Vector{Int},
    s_map::Array{Int,2},gamma::Vector{Float64},
    step::Float64,j::Int,priors::Priors)

    current_beta = beta
    proposed_beta = copy(current_beta)

    current_transformed = log(beta[j])
    proposed_transformed = rand(Normal(current_transformed,step))
    new_beta = exp(proposed_transformed)
    proposed_beta[j] = new_beta
    #new_beta = rand(Normal(current_beta[j],step))
    #proposed_beta[j] = new_beta

    current_lik = log_lik(gamma,M,I_diff,M_beta,current_beta,fail_idx,s_map)
    proposed_lik = log_lik(gamma,M,I_diff,M_beta,proposed_beta,fail_idx,s_map)
    
    log_jump_current = logpdf(
        Normal(
            current_transformed,
            step
        ),
        proposed_transformed
    ) - log(proposed_beta[j])
    
    log_jump_proposed = logpdf(
        Normal(
            proposed_transformed,
            step
        ),
        current_transformed
    ) - log(current_beta[j])
    
    log_lik_ratio = proposed_lik - current_lik
    log_jump_ratio = log_jump_proposed - log_jump_current
    log_prior_ratio = logpdf(priors.beta[j],proposed_beta[j]) -
        logpdf(priors.beta[j],current_beta[j])
    acceptance_ratio = log_lik_ratio + log_jump_ratio + log_prior_ratio

    log_u = log(rand(Uniform(0.0,1.0)))
    accept = log_u <= acceptance_ratio

    if accept
        return_beta = proposed_beta[j]
        #return_risk = proposed_risk
        #main_risk[:,:] .= off_risk
    else
        return_beta = current_beta[j]
        #return_risk = current_risk
    end

    return return_beta,accept
end

"""
    metropolis_beta(...)
CORRECTED implementation for a Metropolis-Hastings update of the linear
risk function coefficient.

This method is for a survival model using a linear risk function.
"""
function metropolis_beta(beta::Float64,M::Array{Float64,2},I_diff::Array{Float64,2},
    stresses::Array{Float64,2},fail_idx::Vector{Int},gamma::Vector{Float64},
    step::Float64,priors::Priors)

    current_beta = beta
    current_transformed = log(beta)
    proposed_transformed = rand(Normal(current_transformed,step))
    proposed_beta = exp(proposed_transformed)
    #proposed_beta = rand(Normal(current_beta,step))

    current_lik = log_lik(gamma,M,I_diff,stresses,current_beta,fail_idx)
    proposed_lik = log_lik(gamma,M,I_diff,stresses,proposed_beta,fail_idx)
    
    log_jump_current = logpdf(
        Normal(
            current_transformed,
            step
        ),
        proposed_transformed
    ) - log(proposed_beta)
    
    log_jump_proposed = logpdf(
        Normal(
            proposed_transformed,
            step
        ),
        current_transformed
    ) - log(current_beta)
    
    log_lik_ratio = proposed_lik - current_lik
    log_jump_ratio = log_jump_proposed - log_jump_current
    log_prior_ratio = logpdf(priors.beta,proposed_beta) -
        logpdf(priors.beta,current_beta)
    acceptance_ratio = log_lik_ratio + log_jump_ratio + log_prior_ratio

    log_u = log(rand(Uniform(0.0,1.0)))
    accept = log_u <= acceptance_ratio

    if accept
        return_beta = proposed_beta
        #return_risk = proposed_risk
        #main_risk[:,:] .= off_risk
    else
        return_beta = current_beta
        #return_risk = current_risk
    end

    return return_beta,accept
end

#############################
# reformulation of corrected MH steps for performance optimization 
"""
    metropolis_gamma(...)
CORRECTED implementation of a Metropolis-Hastings update for a single time domain
spline basis coefficient.

This method is for a survival model using an M-spline risk function
"""
function metropolis_gamma!(
    current_gamma::Vector{Float64},proposed_gamma::Vector{Float64},
    main_I_diff::Vector{Float64},off_I_diff::Vector{Float64},
    main_I_diff_partial::Array{Float64,2},off_I_diff_partial::Array{Float64},
    main_M::Vector{Float64},off_M::Vector{Float64},
    main_M_partial::Array{Float64,2},off_M_partial::Array{Float64,2},
    base_haz_splines::Splines,
    inst_risk::Array{Float64,2},risk_sums::Vector{Float64},
    fail_idx::Vector{Int},step::Float64,idx::Int,
    priors::Priors)
    
    #current_gamma = gamma
    #proposed_gamma = copy(current_gamma)

    proposed_gamma[:] .= current_gamma[:]

    current_transformed = log(current_gamma[idx])
    proposed_transformed = rand(Normal(current_transformed,step))
    new_gamma = exp(proposed_transformed)
    proposed_gamma[idx] = new_gamma

    main_I_diff_partial[:,idx] .= base_haz_splines.I_diff[:,idx] .* current_gamma[idx]
    off_I_diff_partial[:,idx] .= base_haz_splines.I_diff[:,idx] .* proposed_gamma[idx]

    main_M_partial[:,idx] .= base_haz_splines.M[:,idx] .* current_gamma[idx]
    off_M_partial[:,idx] .= base_haz_splines.M[:,idx] .* proposed_gamma[idx]

    main_I_diff[:] .= sum(main_I_diff_partial,dims=2)
    off_I_diff[:] .= sum(off_I_diff_partial,dims=2)

    main_M[:] .= sum(main_M_partial,dims=2)
    off_M[:] .= sum(off_M_partial,dims=2)

    current_lik = log_lik(
        main_I_diff,
        main_M,
        inst_risk,
        risk_sums,
        fail_idx,
    )
    proposed_lik = log_lik(
        off_I_diff,
        off_M,
        inst_risk,
        risk_sums,
        fail_idx,
    )

    log_jump_current = logpdf(
        Normal(
            current_transformed,
            step
        ),
        proposed_transformed
    ) - log(proposed_gamma[idx])
    
    log_jump_proposed = logpdf(
        Normal(
            proposed_transformed,
            step
        ),
        current_transformed
    ) - log(current_gamma[idx])
    
    log_lik_ratio = proposed_lik - current_lik
    log_jump_ratio = log_jump_proposed - log_jump_current
    log_prior_ratio = logpdf(priors.gamma[idx],proposed_gamma[idx]) -
        logpdf(priors.gamma[idx],current_gamma[idx])
    acceptance_ratio = log_lik_ratio + log_jump_ratio + log_prior_ratio

    log_u = log(rand(Uniform(0.0,1.0)))
    accept = log_u <= acceptance_ratio

    if accept
        #return_gamma = proposed_gamma[idx]
        current_gamma[idx] = proposed_gamma[idx]
        main_I_diff_partial[:,idx] .= off_I_diff_partial[:,idx]
        main_I_diff[:] .= off_I_diff[:]

        main_M_partial[:,idx] .= off_M_partial[:,idx]
        main_M[:] .= off_M[:]
    else
        #return_gamma = current_gamma[idx]

        off_I_diff_partial[:,idx] .= main_I_diff_partial[:,idx]
        off_I_diff[:] .= main_I_diff[:]
        off_M_partial[:,idx] .= main_M_partial[:,idx]
        off_M[:] .= main_M[:]
    end

    return accept
end

"""
    metropolis_beta(...)
CORRECTED implementation for a Metropolis-Hastings update of the linear
risk function coefficient.

This method is for a survival model using a linear risk function.
"""
function metropolis_beta!(
    current_beta::Vector{Float64},proposed_beta::Vector{Float64},
    main_M::Vector{Float64},off_M::Vector{Float64},
    main_M_partial::Array{Float64,2},off_M_partial::Array{Float64,2},
    main_inst_risk::Array{Float64,2},main_risk_sums::Vector{Float64},
    off_inst_risk::Array{Float64,2},off_risk_sums::Vector{Float64},
    risk_splines::Splines,
    time_I_diff::Vector{Float64},time_M::Vector{Float64},
    fail_idx::Vector{Int},in_risk_idx::Vector{Vector{Int}},
    s_map::Array{Int,2},step::Float64,idx::Int,priors::Priors)

    #current_beta = beta
    #proposed_beta = copy(current_beta)
    proposed_beta[:] .= current_beta
    #println("Current beta = ",current_beta')
    #println("Jth beta = ",current_beta[idx])
    current_transformed = log(current_beta[idx])
    #println("Current transformed value = ",current_transformed)
    proposed_transformed = rand(Normal(current_transformed,step))
    #println("Proposed transformed value = ",proposed_transformed)
    new_beta = exp(proposed_transformed)
    #println("Converted back to beta = ",new_beta)
    proposed_beta[idx] = new_beta
    #println(proposed_beta[idx])

    #new_beta = rand(Normal(current_beta[idx],step))
    #proposed_beta[idx] = new_beta

    main_M_partial[:,idx] .= risk_splines.M[:,idx] .* current_beta[idx]
    off_M_partial[:,idx] .= risk_splines.M[:,idx] .* proposed_beta[idx]

    main_M[:] .= sum(main_M_partial,dims=2)
    off_M[:] .= sum(off_M_partial,dims=2)

    @inbounds for j in axes(main_inst_risk,2)
        @inbounds for i in axes(main_inst_risk,1)
            main_inst_risk[i,j] = exp(main_M[s_map[i,j]])
            off_inst_risk[i,j] = exp(off_M[s_map[i,j]])
        end
    end

    for i in 1:(length(main_risk_sums)-1)#eachindex(main_risk_sums)
        main_risk_sums[i] = sum(main_inst_risk[i,in_risk_idx[i]])
        off_risk_sums[i] = sum(off_inst_risk[i,in_risk_idx[i]])
    end

    current_lik = log_lik(
        time_I_diff,
        time_M,
        main_inst_risk,
        main_risk_sums,
        fail_idx
    )
    proposed_lik = log_lik(
        time_I_diff,
        time_M,
        off_inst_risk,
        off_risk_sums,
        fail_idx
    )
    #println(current_transformed)
    #println(step)
    #println(proposed_transformed)
    #println(pdf(Normal(current_transformed,step),proposed_transformed))
    #println(proposed_beta[idx])
    #println(current_beta[idx])
    log_jump_current = logpdf(
        Normal(
            current_transformed,
            step
        ),
        proposed_transformed
    ) - log(proposed_beta[idx])
    
    log_jump_proposed = logpdf(
        Normal(
            proposed_transformed,
            step
        ),
        current_transformed
    ) - log(current_beta[idx])
    
    log_lik_ratio = proposed_lik - current_lik
    log_jump_ratio = log_jump_proposed - log_jump_current
    log_prior_ratio = logpdf(priors.beta[idx],proposed_beta[idx]) -
        logpdf(priors.beta[idx],current_beta[idx])
    acceptance_ratio = log_lik_ratio + log_jump_ratio + log_prior_ratio

    log_u = log(rand(Uniform(0.0,1.0)))
    accept = log_u <= acceptance_ratio

    if accept
        current_beta[idx] = proposed_beta[idx]
        main_M_partial[:,idx] .= off_M_partial[:,idx]
        main_M[:] .= off_M[:]
        main_inst_risk[:,:] .= off_inst_risk[:,:]
        main_risk_sums[:] .= off_risk_sums[:]
    else
        off_M_partial[:,idx] .= main_M_partial[:,idx]
        off_M[:] .= main_M[:]
        off_inst_risk[:,:] .= main_inst_risk[:,:]
        off_risk_sums[:] .= main_risk_sums[:]
    end

    return accept
end

# efficient implementations for linear risk function
"""
    metropolis_beta(...)
CORRECTED implementation for a Metropolis-Hastings update of the linear
risk function coefficient.

This method is for a survival model using a linear risk function.
"""
function metropolis_beta!(
    current_beta::Float64,
    main_inst_risk::Array{Float64,2},main_risk_sums::Vector{Float64},
    off_inst_risk::Array{Float64,2},off_risk_sums::Vector{Float64},
    time_I_diff::Vector{Float64},time_M::Vector{Float64},
    fail_idx::Vector{Int},in_risk_idx::Vector{Vector{Int}},
    stresses::Array{Float64,2},step::Float64,priors::Priors)

    current_transformed = log(current_beta)
    proposed_transformed = rand(Normal(current_transformed,step))
    new_beta = exp(proposed_transformed)
    proposed_beta = new_beta

    proposed_beta = rand(Normal(current_beta,step))


    @inbounds for j in axes(main_inst_risk,2)
        @inbounds for i in axes(main_inst_risk,1)
            main_inst_risk[i,j] = exp(current_beta * stresses[i,j])
            off_inst_risk[i,j] = exp(proposed_beta * stresses[i,j])
        end
    end

    for i in 1:(length(main_risk_sums)-1)#eachindex(main_risk_sums)
        main_risk_sums[i] = sum(main_inst_risk[i,in_risk_idx[i]])
        off_risk_sums[i] = sum(off_inst_risk[i,in_risk_idx[i]])
    end

    current_lik = log_lik(
        time_I_diff,
        time_M,
        main_inst_risk,
        main_risk_sums,
        fail_idx
    )
    proposed_lik = log_lik(
        time_I_diff,
        time_M,
        off_inst_risk,
        off_risk_sums,
        fail_idx
    )
    
    log_jump_current = logpdf(
        Normal(
            current_transformed,
            step
        ),
        proposed_transformed
    ) - log(proposed_beta)
    
    log_jump_proposed = logpdf(
        Normal(
            proposed_transformed,
            step
        ),
        current_transformed
    ) - log(current_beta)
    
    log_lik_ratio = proposed_lik - current_lik
    log_jump_ratio = log_jump_proposed - log_jump_current
    log_prior_ratio = logpdf(priors.beta,proposed_beta) -
        logpdf(priors.beta,current_beta)
    acceptance_ratio = log_lik_ratio + log_jump_ratio + log_prior_ratio

    log_u = log(rand(Uniform(0.0,1.0)))
    accept = log_u <= acceptance_ratio

    if accept
        return_beta = proposed_beta
        
        main_inst_risk[:,:] .= off_inst_risk[:,:]
        main_risk_sums[:] .= off_risk_sums[:]
    else
        return_beta = current_beta
        
        off_inst_risk[:,:] .= main_inst_risk[:,:]
        off_risk_sums[:] .= main_risk_sums[:]
    end

    return return_beta,accept
end