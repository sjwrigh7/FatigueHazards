function init(data::StepStressData,spline_order,n_int;risk=:linear)
    interior_knots_grid = collect(
        range(
            start = data.t_norm[2],
            stop = data.t_norm[end-1],
            length = n_int + 2
        )
    )
    interior_knots = interior_knots_grid[2:(end-1)]


    base_haz_splines = generate_splines(spline_order,interior_knots,data.t_norm[1:(end-1)])
    if risk == :linear
        return base_haz_splines
    elseif risk == :splines
        interior_knots_grid = collect(
            range(
                start = minimum(data.s_norm),
                stop = maximum(data.s_norm),
                length = n_int + 2
            )
        )

        interior_knots = interior_knots_grid[2:(end-1)]

        risk_splines = generate_splines(spline_order,interior_knots,sort(unique(data.s_norm)))
        return base_haz_splines,risk_splines
    end
end

function init(data::StepStressData,base_haz_spline_order,base_haz_n_int,risk_spline_order,risk_n_int)
    base_haz_interior_knots_grid = collect(
        range(
            start = data.t_norm[2],
            stop = data.t_norm[end-1],
            length = base_haz_n_int + 2
        )
    )
    base_haz_interior_knots = base_haz_interior_knots_grid[2:(end-1)]


    base_haz_splines = generate_splines(base_haz_spline_order,base_haz_interior_knots,data.t_norm[1:(end-1)])

    risk_interior_knots_grid = collect(
        range(
            start = minimum(data.s_norm),
            stop = maximum(data.s_norm),
            length = risk_n_int + 2
        )
    )

    risk_interior_knots = risk_interior_knots_grid[2:(end-1)]

    risk_splines = generate_splines(risk_spline_order,risk_interior_knots,sort(unique(data.s_norm)))
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

        J = size(data.delta_i,1)

        fail_indic = sum(data.delta_i[2:(end-1),:],dims=2)

        risk_terms = [sum_risk(j,data.s_norm,beta,data.delta_i) for j in 2:(J-1)]
        lik = log_lik(gamma,splines.M,splines.I_diff,J,risk_terms,fail_indic)
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
    n = risk_splines.params.num_basis

    function f(params)
        beta = exp.(params[1:n])
        gamma = exp.(params[(n+1):end])

        J = size(data.delta_i,1)

        fail_indic = sum(data.delta_i[2:(end-1),:],dims=2)

        risk_terms = [sum_risk(j,risk_splines.M,beta,data.in_risk_idx,s_map) for j in 2:(J-1)]
        lik = log_lik(gamma,base_haz_splines.M,base_haz_splines.I_diff,J,risk_terms,fail_indic)
        #log_lik = log_lik_splines(stresses,delta_i,Ts,beta,gamma,M_star,I_star)
        return -lik
    end
    x0 = [0.0 for i in 1:(base_haz_splines.params.num_basis + n)]

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
    println(opt_res.minimizer)
    opt_vals = exp.(opt_res.minimizer)
    return opt_vals
end