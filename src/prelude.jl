function init(data::StepStressData,spline_order,n_int;risk=:linear)
    #interior_knots_grid = collect(
    #    range(
    #        start = data.t_norm[2],
    #        stop = data.t_norm[end-1],
    #        length = n_int + 2
    #    )
    #)
    #interior_knots = interior_knots_grid[2:(end-1)]
    idx_points = Int.(
        floor.(
            collect(
                range(
                    start = 1,
                    stop = length(data.t_norm) - 1,
                    length = n_int + 2
                )
            )
        )
    )[2:(end-1)]

    lower_vals = data.t_norm[idx_points]
    upper_vals = data.t_norm[idx_points .+ 1]

    interior_knots = 0.5 .* (lower_vals .+ upper_vals)


    base_haz_splines = generate_splines(spline_order,interior_knots,data.t_norm[1:(end-1)])

    return base_haz_splines
end

function init(data::StepStressData,base_haz_spline_order,base_haz_n_int,risk_spline_order,risk_n_int)
    base_haz_idx_points = Int.(
        floor.(
            collect(
                range(
                    start = 1,
                    stop = length(data.t_norm) - 1,
                    length = base_haz_n_int + 2
                )
            )
        )
    )[2:(end-1)]

    base_haz_lower_vals = data.t_norm[base_haz_idx_points]
    base_haz_upper_vals = data.t_norm[base_haz_idx_points .+ 1]

    base_haz_interior_knots = 0.5 .* (base_haz_lower_vals .+ base_haz_upper_vals)

    base_haz_splines = generate_splines(
        base_haz_spline_order,
        base_haz_interior_knots,
        data.t_norm[1:(end-1)],
        #data.t_norm[(end-1)] + 1.0
        2.5e8 / data.t_max
    )

    #s_unique = sort(unique(data.s_norm[2:(end-1),:]))
    s_unique = sort(unique(data.s_norm))

    risk_idx_points = Int.(
        floor.(
            collect(
                range(
                    start = 1,
                    stop = length(s_unique) - 1,
                    length = risk_n_int + 2
                )
            )
        )
    )[2:(end-1)]

    risk_lower_vals = s_unique[risk_idx_points]
    risk_upper_vals = s_unique[risk_idx_points .+ 1]

    risk_interior_knots = 0.5 .* (risk_lower_vals .+ risk_upper_vals)


    risk_splines = generate_splines(
        risk_spline_order,
        risk_interior_knots,
        s_unique,
        #s_unique[end] + 1.0
        20000.0 / data.s_max
    )
    return base_haz_splines,risk_splines
end

function map_unique(data::StepStressData)
    s = sort(unique(data.s_norm))
    s_map = Array{Int}(undef,size(data.s_norm))
    for j in axes(s_map,2)
        for i in axes(s_map,1)
            s_map[i,j] = findfirst(x -> x == data.s_norm[i,j],s)
        end
    end
    return s_map
end

function init_sampler(data::StepStressData,splines::Splines)
    opt = opt_lik(data,splines)
    # TODO
    # add step size solver
    # solve step size
end

# define likelihood maximization function
function opt_lik(data::StepStressData,splines::Splines)
    function f(params)
        beta = exp(params[1])
        gamma = exp.(params[2:end])

        #fail_indic = sum(data.delta_i[2:(end-1),:],dims=2)

        #risk_terms = [sum_risk(j,data.s_norm,beta,data.delta_i) for j in 2:(J-1)]
        lik = log_lik(
            gamma,
            splines.M,
            splines.I_diff,
            data.s_norm,
            beta,
            data.fail_idx
        )
        #log_lik = log_lik_splines(stresses,delta_i,Ts,beta,gamma,M_star,I_star)
        return -lik
    end
    x0 = [0.0 for i in 1:(splines.params.num_basis + 1)]

    opt_res = Optim.optimize(
        f,
        x0,
        Optim.LBFGS(),
        Optim.Options(
            store_trace=true,
            extended_trace=true,
        );
        autodiff = ADTypes.AutoForwardDiff()
    )

    opt_vals = exp.(opt_res.minimizer)
    return opt_vals
end

# define likelihood maximization function
function opt_lik(data::StepStressData,base_haz_splines::Splines,risk_splines::Splines,s_map::Array{Int,2})
    n_risk = risk_splines.params.num_basis
    n_base = base_haz_splines.params.num_basis
    function f(params)
        beta = exp.(params[1:n_risk])
        gamma = exp.(params[(n_risk+1):end])

        lik = log_lik(
            gamma,
            base_haz_splines.M,
            base_haz_splines.I_diff,
            risk_splines.M,
            beta,
            data.fail_idx,
            s_map
        )
        #log_lik = log_lik_splines(stresses,delta_i,Ts,beta,gamma,M_star,I_star)
        return -lik
    end
    #x0 = [0.0 for i in 1:(base_haz_splines.params.num_basis + n)]
    
    n_opt = 150
    opt_mins = Vector{Float64}(undef,n_opt)
    opt_inps = Array{Float64}(undef,n_opt,n_risk+n_base)

    x0_bounds = [(-3.0,3.0) for _ in 1:(n_base+n_risk)]
    x0_doe = LHCoptim(n_opt,n_risk+n_base,5)
    x0_doe = scaleLHC(x0_doe[1],x0_bounds)

    for i in 1:n_opt
        opt_res = Optim.optimize(
            f,
            x0_doe[i,:],
            Optim.LBFGS(),
            Optim.Options(
                store_trace=true,
                extended_trace=true,
            );
            autodiff = ADTypes.AutoForwardDiff()
        )
        opt_mins[i] = opt_res.minimum
        opt_inps[i,:] = exp.(opt_res.minimizer)
    end
    real_idx = isfinite.(opt_mins)
    opt_mins = opt_mins[real_idx]
    opt_inps = opt_inps[real_idx,:]

    min_found,min_idx = findmin(opt_mins)
    opt_vals = opt_inps[min_idx,:]
    println(opt_mins)
    println("Maximum log-likelihood found = $(-min_found)")
    println("Corresponding inputs:")
    println("β = $(opt_vals[1:n_risk])")
    println("γ = $(opt_vals[(n_risk+1):end])")
    
    return opt_vals
end