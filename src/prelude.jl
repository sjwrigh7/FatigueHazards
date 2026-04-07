function init(data::StepStressData,spline_order,n_int)
    interior_knots_grid = collect(
        range(
            start = data.t_norm[2],
            stop = data.t_norm[end-1],
            length = n_int + 2
        )
    )
    interior_knots = interior_knots_grid[2:(end-1)]

    splines = generate_splines(spline_order,interior_knots,data.t_norm[1:(end-1)])

    return splines
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
        lik = log_lik(data.s_norm,gamma,splines.M,splines.I_diff,J,risk_terms,fail_indic)
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

