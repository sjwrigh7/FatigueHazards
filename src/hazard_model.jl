"""
    log_lik(x::)
Log-likelihood of the hazard model for M/I spline baseline hazards and linear risk.
"""
function log_lik(x,gamma,M,I_diff,J,risk_terms,fail_indic)
    delta_cumulative_base_hazards = I_diff * gamma
    term1 = - sum(risk_terms .* delta_cumulative_base_hazards)

    base_hazards = M[2:end,:] * gamma
    term2 = sum(log.(base_hazards .* risk_terms) .^ fail_indic)

    lik = term1 + term2

    return lik
end

"""
    sum_risk(j,x,beta,delta_i)
Calculates the sum of linear risk function values over i=1,...,n specimens that have not yet failed
"""
function sum_risk(j::Int,x::Array{Float64,2},beta::Float64,delta_i::Array{Int,2})
    idx = in_risk(j,delta_i)

    sum_risk = 0.0
    @inbounds for i in idx
        s = x[j,i]
        sum_risk += exp(s * beta)
    end
    return sum_risk
end

function sum_risk(j::Int,x::Array{Float64,2},beta::Float64,in_risk_idx::Vector{Vector{Int}})
    idx = in_risk_idx[j]

    sum_risk = 0.0
    @inbounds for i in idx
        s = x[j,i]
        sum_risk += exp(s * beta)
    end
    return sum_risk
end

function sum_risk(j::Int,x::Array{Float64,2},beta::ForwardDiff.Dual,delta_i::Array{Int,2})
    idx = in_risk(j,delta_i)

    sum_risk = 0.0
    @inbounds for i in idx
        s = x[j,i]
        sum_risk += exp(s * beta)
    end
    return sum_risk
end

function sum_risk(j::Int,x::Array{Float64,2},beta::ForwardDiff.Dual,in_risk_idx::Vector{Vector{Int}})
    idx = in_risk_idx[j]

    sum_risk = 0.0
    @inbounds for i in idx
        s = x[j,i]
        sum_risk += exp(s * beta)
    end
    return sum_risk
end

"""
    in_risk(j,delta_i)
Evaluates the set of specimens that have not yet failed at index j of the time vector
"""
function in_risk(j,delta_i)
    #idx = findall(x -> x == 0, vec(sum(delta_i[1:(j-1),:],dims=1)))
    idx = findall(x -> x == 0, vec(sum(delta_i[1:(j-1),:],dims=1)))
    return idx
end

"""
    cumulative_hazard()
Calculates the cumulative hazard function over a time grid with corresponding values of 
time varying covariates for a linear risk function
"""
function cumulative_hazard(x::Vector{Float64},beta::Float64,
    gamma::Vector{Float64},I_diff::Array{Float64,2})

    risk = exp.(x * beta)

    c_hazard = cumsum((I_diff * gamma) .* risk)

    return c_hazard
end

"""
Evaluates the cumulative hazard function over a time grid with corresponding values of
time varying covariates with a linear risk function, with precomputed sum of I spline basis functions
"""
function cumulative_hazard(x::Vector{Float64},beta::Float64,
    I_diff::Array{Float64,2})

    risk = exp.(x * beta)

    c_hazard = cumsum(I_diff .* risk)

    return c_hazard
end

"""
Evaluates the cumulative hazard function over a time grid with precomputed sum of I spline basis functions
and a precomputed risk
"""
function cumulative_hazard(risk::Vector{Float64},I_diff::Vector{Float64})

    c_hazard = cumsum(I_diff .* risk)

    return c_hazard
end

"""
Evaluates the cumulative hazard function over a time grid with precomputed sum of I spline basis functions
and a precomputed risk
"""
function cumulative_hazard_scalar(risk::Vector{Float64},I_diff::Vector{Float64})

    c_hazard = sum(I_diff .* risk)

    return c_hazard
end

"""
Evaluates the cumulative hazard function over a time grid with precomputed sum of I spline basis functions
and a precomputed risk
"""
function cumulative_hazard_scalar(risk::SubArray{Float64},I_diff::SubArray{Float64})

    c_hazard = sum(I_diff .* risk)

    return c_hazard
end

"""
Evaluates the cumulative hazard function over a time grid with precomputed sum of I spline basis functions
and a precomputed risk
"""
function cumulative_hazard_scalar(mul_blank::SubArray{Float64},risk::SubArray{Float64},I_diff::SubArray{Float64})

    mul_blank .= risk .* I_diff
    c_hazard = sum(mul_blank)

    return c_hazard
end

"""
    survival
Evaluates the hazard model's survival function over a time grid with corresponding values of 
time varying covariates for a linear risk function

"""
function survival(x::Vector{Float64},beta::Float64,
    gamma::Vector{Float64},I_diff::Array{Float64})

    c_hazard = cumulative_hazard(
        x,
        beta,
        gamma,
        I_diff
    )

    survival = exp.(-c_hazard)
end

"""
    cumulative_hazard_constx(x,beta,gamma,I)
Evaluates the cumulative hazard function over a matrix of different time and covariate values,
where covariates are constant over time, using a linear risk function
"""
function cumulative_hazard_constx(x::Vector{Float64},beta::Float64,
    gamma::Vector{Float64},I::Array{Float64,2})

    c_hazard = exp.((x * beta)') .* (I * gamma)

    return c_hazard
end

"""
    survival_constx(x,beta,gamma,I)
Evaluates the hazard model's survival function over a matrix of different time and covariate values,
where covariates are constant over time, using a linear risk function.
"""
function survival_constx(x::Vector{Float64},beta::Float64,
    gamma::Vector{Float64},I::Array{Float64,2})

    c_hazard = cumulative_hazard_constx(
        x,
        beta,
        gamma,
        I
    )

    survival = exp.(-c_hazard)

    return survival
end

"""
Evaluates the log hazard function for specified M basis functions, with a linear risk function
"""
function log_hazard(x::Float64,beta::Float64,gamma::Vector{Float64},M::Vector{Float64})
    risk = exp(x * beta)

    hazard = log(
        M' * gamma * risk
    )

    return hazard
end

"""
Evaluates the log hazard function for specified M basis functions, with a precomputed risk function
"""
function log_hazard(risk::Float64,gamma::Vector{Float64},M::Vector{Float64})
    hazard = log(
        M' * gamma * risk
    )

    return hazard
end

"""
Evaluates the log hazard function for precomputed sum of basis functions of M, with a linear risk function
"""
function log_hazard(x::Float64,beta::Float64,M::Float64)
    risk = exp(x * beta)

    hazard = log(
        M * risk
    )

    return hazard
end

"""
Evaluates the log hazard function for precomputed sum of basis functions of M, and a precomputed
risk function
"""
function log_hazard(risk::Float64,M::Float64)

    hazard = log(
        min(
            M * risk,
            prevfloat(floatmax(Float64))
        )
    )

    return hazard
end

"""
    density()
Evaluates the hazard model density function at point t, for a grid of time values with corresponding
time varying covariate values, using a linear risk term.

The time grid and covariate vectors must me constructed such that t occurrs in the time grid

I_diff is the matrix of differences between time grid points of the I basis functions
M is the matrix of M basis functions over the time grid

the time grid and x grid both have a value of 0 at index 1
"""
function log_density(t::Float64,x::Vector{Float64},time_grid::Vector{Float64},
    beta::Float64,gamma::Vector{Float64},I_diff::Array{Float64,2},M::Array{Float64,2})

    target_idx = findfirst(y -> isapprox(y,t), time_grid)

    # vector of risk function values at each time point up to t
    risk = exp.(beta .* x[1:target_idx])

    # cumulative hazard function up to time t
    # since I_diff is the differences between index i and i-1 over length of time_grid
    # and has length n-1, index j-1 of I_diff corresponds to index j in time
    cumulative = cumulative_hazard(
        x[2:target_idx],
        beta,
        gamma,
        I_diff[1:(target_idx - 1),:]
    )[end]

    # instantaneous hazard function at time t
    instant = log_hazard(
        risk[target_idx],
        gamma,
        M[target_idx,:]
    )

    density = cumulative * instant

    return density
end

"""
Evaluates the log density of the hazard model via precomputed sums of M and I_diff bases
using a linear risk function
"""
function log_density(t::Float64,x::Vector{Float64},time_grid::Vector{Float64},
    beta::Float64,I_diff::Vector{Float64},M::Vector{Float64})

    target_idx = findfirst(y -> isapprox(y,t), time_grid)

    # vector of risk function values at each time point up to t
    risk = exp.(beta .* x[1:target_idx])

    # cumulative hazard function up to time t
    # since I_diff is the differences between index i and i-1 over length of time_grid
    # and has length n-1, index j-1 of I_diff corresponds to index j in time
    cumulative = cumulative_hazard(
        risk[1:target_idx],
        I_diff[1:(target_idx - 1)]
    )

    # instantaneous hazard function at time t
    instant = log_hazard(
        risk[target_idx],
        M[target_idx]
    )

    density = cumulative * instant

    return density
end

"""
Evaluates the log density of the hazard model via precomputed sums of M and I_diff bases
and a precomputed linear risk function
"""
function log_density(t::Float64,time_grid::Vector{Float64},
    risk::Vector{Float64},I_diff::Vector{Float64},M::Vector{Float64},target_idx::Int)

    # cumulative hazard function up to time t
    # since I_diff is the differences between index i and i-1 over length of time_grid
    # and has length n-1, index j-1 of I_diff corresponds to index j in time
    cumulative = cumulative_hazard_scalar(
        view(risk,2:target_idx),
        view(I_diff,1:(target_idx - 1))
    )

    # instantaneous hazard function at time t
    instant = log_hazard(
        risk[target_idx],
        M[target_idx]
    )

    density = -cumulative + instant
    #println(cumulative)
    return density
end

"""
Evaluates the log density of the hazard model via precomputed sums of M and I_diff bases
and a precomputed linear risk function
"""
function log_density!(mul_blank::Vector{Float64},time_grid::Vector{Float64},
    risk::Vector{Float64},I_diff::Vector{Float64},M::Vector{Float64},target_idx::Int)

    # cumulative hazard function up to time t
    # since I_diff is the differences between index i and i-1 over length of time_grid
    # and has length n-1, index j-1 of I_diff corresponds to index j in time
    cumulative = cumulative_hazard_scalar(
        view(mul_blank,1:(target_idx - 1)),
        view(risk,2:target_idx),
        view(I_diff,1:(target_idx - 1))
    )

    # instantaneous hazard function at time t
    instant = log_hazard(
        risk[target_idx],
        M[target_idx]
    )

    density = -cumulative + instant
    #println(cumulative)
    return density
end

"""
design initialization function for sampling failure time
"""
function init_design(design::StepStressTest,spline_design::SplineDesign)
    # init time grid from 0 to maximum failure time in original data
    t_grid = collect(range(
        start = 0.0,
        step = design.n,
        stop = 1.0
    ))
    #println(design.n)
    # init stress grid over length of time grid
    stress_grid = vcat(
        0.0,
        collect(range(
            start = design.s0,
            step = design.ds,
            length = length(t_grid)-1
        ))
    )

    if design.ds < 0.0
        target_idx = findlast(x -> x > 0.0,stress_grid)
        #println(stress_grid[1:5])
        #println(target_idx)
        target_stress = stress_grid[target_idx]
        stress_grid[target_idx:end] .= target_stress
    end

    splines = generate_splines(
        spline_design.k,
        spline_design.interior_knots,
        t_grid
    )

    return splines,stress_grid,t_grid
end

"""
function to find k, the first index in time grid that would exceed the
random failure chance, u
"""
function find_k(gamma,I_diff,risk_terms)
    # generate random uniform chance of failure
    u = rand(Uniform(0.0,1.0))
    # take log for efficient comparisons
    log_u = -log(u)

    # evaluate the survival function values over the time grid
    survival_vals = cumsum((I_diff * gamma) .* risk_terms)
    
    # find failure index, k
    k = findfirst(x -> x > log_u, survival_vals)

    #println(k)
    #println(isnothing(k))
    # safety check to make sure 
    if isnothing(k)
        k = 1
        prev_surv = 0.0
    elseif k == 1
        prev_surv = 0.0
    else
        prev_surv = survival_vals[k-1]
    end

    return log_u,k,prev_surv,survival_vals[k]
end

"""
recursive binary search function to find actual failure time for random sample
"""
function bisect_recurse(target,t1,t2,S1,S2,spline_params,gamma,tol)
    # evaluate the center point between the lower and upper bounds
    t_center = 0.5 * (t1 + t2)
    I_center = vec(eval_i_spline(
        spline_params.design.k,
        spline_params.num_basis,
        spline_params.knot_grid,
        [t_center]
    ))
    S_center = sum(I_center .* gamma)
    
    # update the lower or upper bounds to use the current center value based on the target
    if abs((S1 - S2) / target) > tol
        if S_center < target
            t1 = t_center
            S1 = S_center
        else
            t2 = t_center
            S2 = S_center
        end

        t_center = bisect_recurse(
            target,
            t1,
            t2,
            S1,
            S2,
            spline_params,
            gamma,
            tol
        )
    end
    return t_center#,S_center
end

"""

"""
function sample_t(gamma,splines,risk_terms,t_grid,tol=1e-6)
    # draw random uniform sample and find the first index, k, in the time grid
    # that corresponds to the survival function value exceeding u
    # k is evaluated based on the cumulative hazard function, using I_diff, meaning
    # the index k corresponds to time at k+1 in the time grid
    log_u,k,S1,S2 = find_k(
        gamma,
        splines.I_diff,
        risk_terms
    )
    # I basis functions at time gid value just before failure
    # evaluated by time grid so index is k + 1
    I_prev = splines.I[k,:]

    # normalize the log_u random draw so only the I basis comparison is needed
    target = (log_u - S1) / risk_terms[k] +
    sum(vec(gamma) .* vec(I_prev))

    # normalize the S2 value in the same way
    I_next = (S2 - S1) / risk_terms[k] + 
    sum(vec(gamma) .* vec(I_prev))

    t_sample = bisect_recurse(
        target,
        t_grid[k],
        t_grid[k+1],
        0.0,
        I_next,
        splines.params,
        gamma,
        tol
    )

    return t_sample,k
end