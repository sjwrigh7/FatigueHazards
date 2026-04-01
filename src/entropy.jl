function eval_entropy(design::StepStressTest,data::StepStressData,posterior_iid::PosteriorIID,
    spline_design::SplineDesign,n_sim_outer::Int,n_sim_inner;results=:scalar)

    log_cond,log_marg = _eval_entropy(
        design,
        data,
        posterior_iid,
        spline_design,
        n_sim_outer,
        n_sim_inner
    )


    log_marg = eval_log_marg(log_marg; res=:full)

    if results == :full
        return log_cond,log_marg
    else
        if results == :vector
            return log_cond, log_marg[end,:]
        elseif results == :scalar
            ent = mean(log_cond) - mean(log_marg[end,:])
            return ent
        end
    end
end

function _eval_entropy(design::StepStressTest,data::StepStressData,posterior_iid::PosteriorIID,
    spline_design::SplineDesign,n_sim_outer::Int,n_sim_inner)

    sample_avail = length(posterior_iid.beta)

    design_norm = StepStressTest(
        design.s0 / data.s_max,
        design.ds / data.s_max,
        design.n / data.t_max
    )

    splines,stress_grid,t_grid = init_design(design_norm,spline_design)

    if n_sim_outer > sample_avail
        @warn "The number of desired outer Monte Carlo samples is $n_sim_outer...
        but only $(sample_avail) realizations of β and γ are available."
        n_sim_outer = min(n_sim_outer,sample_avail)
        println("Reducing the number of outer Monte Carlo simulation points to $n_sim_outer")
    end
    if n_sim_inner > sample_avail
        @warn "The number of desired outer Monte Carlo samples is $n_sim_inner...
        but only $(sample_avail) realizations of β and γ are available."
        n_sim_inner = min(n_sim_inner,sample_avail)
        println("Reducing the number of outer Monte Carlo simulation points to $n_sim_inner")
    end
    
    outer_idx = sample(1:sample_avail,n_sim_outer,replace=false)

    beta_outer = posterior_iid.beta[outer_idx]
    gamma_outer = posterior_iid.gamma[outer_idx,:]

    thin_val = Int(floor(sample_avail / n_sim_inner))
    inner_idx = (
        collect(
            range(
                start = 1,
                step = thin_val,
                length = n_sim_inner
            )
        )
    )


    beta_inner = posterior_iid.beta[inner_idx]
    gamma_inner = posterior_iid.gamma[inner_idx,:]

    t_samples = Vector{Float64}(undef,n_sim_outer)
    ks = Vector{Int}(undef,n_sim_outer)

    @showprogress "Sampling failure time..." for i in 1:n_sim_outer
        # pre calculate risk terms over time grid
        risk_terms = exp.(stress_grid[2:end] .* beta_outer[i])
        #println(r_gamma[i,:])
        t,k = sample_t(
            gamma_outer[i,:],
            splines,
            risk_terms,
            t_grid,
            1e-6
        )
        t_samples[i] = t
        ks[i] = k
    end



    combined_time,combined_stress,time_grid_idx,param_idx = merge_grids(
        t_grid,
        stress_grid,
        t_samples,
        ks
    )

    update_x!(splines,combined_time)
    ####################
    # try precomputing all vals
    risk_outer = [exp.(combined_stress * beta_outer[i]) for i in eachindex(beta_outer)]
    I_diff_outer = [splines.I_diff * gamma_outer[i,:] for i in axes(gamma_outer,1)]
    M_outer = [splines.M * gamma_outer[i,:] for i in axes(gamma_outer,1)]

    risk_inner = [exp.(combined_stress * beta_inner[j]) for j in eachindex(beta_inner)]
    I_diff_inner = [splines.I_diff * gamma_inner[j,:] for j in axes(gamma_inner,1)]
    M_inner  = [splines.M * gamma_inner[j,:] for j in axes(gamma_inner,1)]
    ####################

    log_cond = Vector{Float64}(undef,n_sim_outer)
    log_marg = Array{Float64}(undef,n_sim_inner,n_sim_outer)
    mul_blank = Vector{Float64}(undef,maximum(time_grid_idx))

    #=
    println("risk_random = ",size(risk_random))
    println("combined time = ",size(combined_time))
    println("random draws = ",length(r_idx))
    println("new idx = ",length(new_idx))
    println(maximum(new_idx))
    println("merged idx = ",length(merged_idx))
    println(maximum(merged_idx))
    println("n sim = ",n_sim)
    println("I diff rand = ",size(I_diff_random))
    =#

    @showprogress "Computing entropy..." for i in 1:n_sim_outer
        log_cond[i] = log_density!(
            mul_blank,
            combined_time,
            risk_outer[param_idx[i]],
            I_diff_outer[param_idx[i]],
            M_outer[param_idx[i]],
            time_grid_idx[i]
        )
        #log_marg_inner = 0.0
        for j in 1:n_sim_inner
            log_marg[j,i] = log_density!(
                mul_blank,
                combined_time,
                risk_inner[j],
                I_diff_inner[j],
                M_inner[j],
                time_grid_idx[i]
            )
        end
    end

    return log_cond,log_marg
end

function eval_inner_chain(log_marg_chain;band=0)
    n_inner = size(log_marg_chain,1)
    r_idx = sample(1:n_inner,n_inner,replace=false)

    marg = exp.(log_marg_chain[r_idx,:])

    running_means = cumsum(marg,dims=1) ./ (1:n_inner)

    if band == 0
        band = max(50,Int(floor(0.05 * n_inner)))
    end

    println(band)
    band_vars = Array{Float64}(undef,(n_inner - 2 * band),size(running_means,2))

    for j in axes(band_vars,2)
        for i in axes(band_vars,1)
            temp = running_means[i:(i + 2 * band - 1),j]
            band_vars[i,j] = std(temp)
        end
    end

    _,best_idx = findmin(band_vars[end,:])
    _,worst_idx = findmax(band_vars[end,:])

    p1 = plot(
        1:size(running_means,1),
        running_means[:,best_idx] .- running_means[end,best_idx],
        label="MC Sample Mean"
    )
    plot!(
        collect(1:size(band_vars,1)) .+ band,
        band_vars[:,best_idx],
        label = "MC Band Variance"
    )
    title!("Convergence of Best Chain")
    xlabel!("MC Iteration")
    ylabel!("Sample Value")

    p2 = plot(
        1:size(running_means,1),
        running_means[:,worst_idx] .- running_means[end,worst_idx],
        label="MC Sample Mean"
    )
    plot!(
        collect(1:size(band_vars,1)) .+ (band),
        band_vars[:,worst_idx],
        label = "MC Band Variance"
    )
    title!("Convergence of Worst Chain")
    xlabel!("MC Iteration")
    ylabel!("Sample Value")

    println(best_idx)
    println(worst_idx)
    #display(p1)
    display(p1)
end

function eval_log_marg(log_marg::Array{Float64,2};res=:scalar)
    results = _eval_log_marg(log_marg,res)

    if res == :full
        return results
    elseif res == :scalar
        return results[end,:]
    end
end

function _eval_log_marg(log_marg::Array{Float64,2},res)
    log_marg_expect = similar(log_marg)

    @inbounds for i in axes(log_marg,2)
        log_marg_expect[:,i] .= log_sum_exp(log_marg[:,i];res=res) .- log.(collect(1:size(log_marg,1)))
    end

    return log_marg_expect
end

function log_sum_exp(p::Vector{Float64};res=:scalar)
    results = _log_sum_exp(p)

    if res == :full
        return results
    elseif res == :scalar
        return results[end]
    end
end

function _log_sum_exp(p::Vector{Float64})
    a = maximum(p)
    if !isfinite(a)
        return a
    end

    s = similar(p)

    @inbounds for i in eachindex(p)
        s[i] = exp(p[i] - a)
    end

    return a .+ log.(cumsum(s))
end