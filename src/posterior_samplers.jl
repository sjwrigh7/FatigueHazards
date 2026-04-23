# more efficient metropolis function
function metropolis_gamma(gamma::Vector{Float64},M::Array{Float64,2},I_diff::Array{Float64,2},
    stresses::Array{Float64,2},J::Int,risk_terms::Vector{Float64},fail_indic::Array{Int,2},step::Float64,j::Int)
    
    current_gamma = gamma
    proposed_gamma = copy(current_gamma)

    current_transformed = log(gamma[j])
    proposed_transformed = rand(Normal(current_transformed,step))
    new_gamma = exp(proposed_transformed)
    proposed_gamma[j] = new_gamma

    #current_lik = log_lik_splines(stresses,delta_i,T,beta,current_gamma,M,I)
    #proposed_lik = log_lik_splines(stresses,delta_i,T,beta,proposed_gamma,M,I)

    current_lik = log_lik(current_gamma,M,I_diff,J,risk_terms,fail_indic)
    proposed_lik = log_lik(proposed_gamma,M,I_diff,J,risk_terms,fail_indic)

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
    
    acceptance_ratio = min(
        exp(
            log_lik_ratio + log_jump_ratio
        ),
        1.0
    )

    u = rand(Uniform(0.0,1.0))
    accept = u <= acceptance_ratio

    if accept
        return_gamma = proposed_gamma[j]
    else
        return_gamma = current_gamma[j]
    end

    return return_gamma,accept
end

# more efficient metropolis function
function metropolis_beta(beta,M,I_diff,stresses,fail_indic,delta_i,J,gamma,step)
    current_beta = beta
    current_transformed = log(beta)
    proposed_transformed = rand(Normal(current_transformed,step))
    proposed_beta = exp(proposed_transformed)

    #proposed_beta = current_beta + rand(Normal(0.0,step))
    current_risk = [sum_risk(j,stresses,current_beta,delta_i) for j in 2:(J-1)]
    proposed_risk = [sum_risk(j,stresses,proposed_beta,delta_i) for j in 2:(J-1)]

    #current_lik = log_lik_splines(stresses,delta_i,T,current_beta,gamma,M,I)
    #proposed_lik = log_lik_splines(stresses,delta_i,T,proposed_beta,gamma,M,I)

    current_lik = log_lik(gamma,M,I_diff,J,current_risk,fail_indic)
    proposed_lik = log_lik(gamma,M,I_diff,J,proposed_risk,fail_indic)
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
    
    acceptance_ratio = min(
        exp(
            log_lik_ratio + log_jump_ratio
        ),
        1.0
    )

    u = rand(Uniform(0.0,1.0))
    accept = u <= acceptance_ratio

    if accept
        return_beta = proposed_beta
        return_risk = proposed_risk
    else
        return_beta = current_beta
        return_risk = current_risk
    end

    return return_beta,accept,return_risk
end

# more efficient metropolis function
function metropolis_beta!(main_risk::Vector{Float64},off_risk::Vector{Float64},beta::Float64,
    M::Array{Float64,2},I_diff::Array{Float64,2},stresses::Array{Float64,2},fail_indic::Array{Int,2},
    in_risk_idx::Vector{Vector{Int}},J::Int,gamma::Vector{Float64},step::Float64)

    current_beta = beta
    current_transformed = log(beta)
    proposed_transformed = rand(Normal(current_transformed,step))
    proposed_beta = exp(proposed_transformed)

    @inbounds for j in eachindex(main_risk)
        main_risk[j] = sum_risk(j+1,stresses,current_beta,in_risk_idx)
        off_risk[j] = sum_risk(j+1,stresses,proposed_beta,in_risk_idx)
    end

    current_lik = log_lik(gamma,M,I_diff,J,main_risk,fail_indic)
    proposed_lik = log_lik(gamma,M,I_diff,J,off_risk,fail_indic)
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
    
    acceptance_ratio = min(
        exp(
            log_lik_ratio + log_jump_ratio
        ),
        1.0
    )

    u = rand(Uniform(0.0,1.0))
    accept = u <= acceptance_ratio

    if accept
        return_beta = proposed_beta
        #return_risk = proposed_risk
        main_risk[:] .= off_risk
    else
        return_beta = current_beta
        #return_risk = current_risk
    end

    return return_beta,accept
end

# more efficient metropolis function
function metropolis_beta!(main_risk::Vector{Float64},off_risk::Vector{Float64},beta::Vector{Float64},
    M::Array{Float64,2},I_diff::Array{Float64,2},M_beta::Array{Float64,2},fail_indic::Array{Int,2},
    in_risk_idx::Vector{Vector{Int}},s_map::Array{Int,2},J::Int,gamma::Vector{Float64},step::Float64,j::Int)

    #current_beta = beta
    #current_transformed = log(beta)
    #proposed_transformed = rand(Normal(current_transformed,step))
    #proposed_beta = exp(proposed_transformed)

    current_beta = beta
    proposed_beta = copy(current_beta)

    current_transformed = log(beta[j])
    proposed_transformed = rand(Normal(current_transformed,step))
    new_beta = exp(proposed_transformed)
    proposed_beta[j] = new_beta

    @inbounds for k in eachindex(main_risk)
        main_risk[k] = sum_risk(k+1,M_beta,current_beta,in_risk_idx,s_map)
        off_risk[k] = sum_risk(k+1,M_beta,proposed_beta,in_risk_idx,s_map)
    end

    current_lik = log_lik(gamma,M,I_diff,J,main_risk,fail_indic)
    proposed_lik = log_lik(gamma,M,I_diff,J,off_risk,fail_indic)
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
    
    acceptance_ratio = min(
        exp(
            log_lik_ratio + log_jump_ratio
        ),
        1.0
    )

    u = rand(Uniform(0.0,1.0))
    accept = u <= acceptance_ratio

    if accept
        return_beta = proposed_beta[j]
        #return_risk = proposed_risk
        main_risk[:] .= off_risk
    else
        return_beta = current_beta[j]
        #return_risk = current_risk
    end

    return return_beta,accept
end