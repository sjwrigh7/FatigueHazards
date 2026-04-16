function init_opt_design(s_min=1e3,s_max=2e4,ds_min=-1e4,ds_max=1e4,n_min=1e3,n_max=1e7;reduce=false,n_const=5e3)
    if reduce
        design = OptDesignReduced(
            s_min,
            s_max,
            ds_min,
            ds_max,
            n_const
        )
    else
        design = OptDesign(
            s_min,
            s_max,
            ds_min,
            ds_max,
            n_min,
            n_max
        )
    end
    return design
end

function init_data(
    design::OptDesign,
    samples::PosteriorIID,
    data::StepStressData,
    splines::Splines;
    n_init=30,
    n_rep=7,
    n_inner=2500,
    n_outer=2500,
)

    doe_bounds = [
        (design.s_min,design.s_max),
        (design.ds_min,design.ds_max),
        (design.n_min,design.n_max)
    ]
    doe = LHCoptim(n_init,3,100)
    doe = scaleLHC(doe[1],doe_bounds)

    lower_bounds = [p[1] for p in doe_bounds]
    upper_bounds = [p[2] for p in doe_bounds]

    doe_norm = (doe .- lower_bounds') ./ (upper_bounds' .- lower_bounds')

    doe_resp = Array{Float64}(undef,n_init,n_rep)

    @showprogress "Solving initial design..." for i in 1:n_init
        temp_design = StepStressTest(
            doe[i,1],
            doe[i,2],
            doe[i,3]
        )

        for j in 1:n_rep
            ent_val = eval_entropy(
                temp_design,
                data,
                samples,
                splines.params.design,
                n_outer,
                n_inner,
                results=:scalar
            )

            doe_resp[i,j] = ent_val
        end
    end

    return lower_bounds,upper_bounds,doe_norm,doe_resp
end

function init_data(
    design::OptDesignReduced,
    samples::PosteriorIID,
    data::StepStressData,
    splines::Splines;
    n_init=15,
    n_rep=7,
    n_inner=2500,
    n_outer=2500
)

    doe_bounds = [
        (design.s_min,design.s_max),
        (design.ds_min,design.ds_max),
    ]
    doe = LHCoptim(n_init,2,100)
    doe = scaleLHC(doe[1],doe_bounds)

    lower_bounds = [p[1] for p in doe_bounds]
    upper_bounds = [p[2] for p in doe_bounds]

    doe_norm = (doe .- lower_bounds') ./ (upper_bounds' .- lower_bounds')

    doe_resp = Array{Float64}(undef,n_init,n_rep)

    @showprogress "Solving initial design..." for i in 1:n_init
        temp_design = StepStressTest(
            doe[i,1],
            doe[i,2],
            design.n_const
        )
        for j in 1:n_rep
            ent_val = eval_entropy(
                temp_design,
                data,
                samples,
                splines.params.design,
                n_outer,
                n_inner,
                results=:scalar
            )

            doe_resp[i,j] = ent_val
        end
    end

    return lower_bounds,upper_bounds,doe_norm,doe_resp
end

function optimize_design(
    samples::PosteriorIID,
    data::StepStressData,
    splines::Splines,
    n_opt::Int;
    s_min=1e3,
    s_max=2e4,
    ds_min=-1e4,
    ds_max=1e4,
    n_min=1e3,
    n_max=1e7,
    n_const=5e3,
    reduce=false,
    n_init=0,
    n_rep=7,
    n_use=3,
    n_inner=2500,
    n_outer=2500,
    n_mcmc=20000,
    pop_size=5000,
    max_time=1.5

)
    design = init_opt_design(
        s_min,
        s_max,
        ds_min,
        ds_max,
        n_min,
        n_max;
        reduce=reduce,
        n_const=n_const
    )

    if n_init == 0
        if reduce
            n_init = 15
        else
            n_init = 30
        end
    end
    println(n_init)

    lower_bounds,upper_bounds,doe_norm,doe_resp = init_data(
        design,
        samples,
        data,
        splines;
        n_init=n_init,
        n_rep=n_rep,
        n_inner=n_inner,
        n_outer=n_outer
    )

    opt_bounds = [(0.0,1.0) for i in eachindex(lower_bounds)]

    mdl = ElasticGPE(
        length(opt_bounds),
        mean = MeanConst(0.5),
        kernel = SE(repeat([-1.50],length(opt_bounds)),-2.0),
        logNoise = -3.0,
        capacity = 3000
    )

    # set priors for GP
    set_priors!(
        mdl.mean,
        [Normal(0.0,1.0)]
    )
    set_priors!(
        mdl.logNoise,
        [Normal(-3.0,3.0)]
    )
    set_priors!(
        mdl.kernel,
        vcat(
            repeat(
                [Normal(-1.5,3.0)],
                length(opt_bounds)
            ),
            Normal(0.0,3.0)
        )
    )

    for i in 1:n_use
        println("Initial DOE Shannon entropy = ")
        println(vec(sort(doe_resp,dims=2)[:,i]))
        append!(mdl,permutedims(doe_norm),vec(sort(doe_resp,dims=2,rev=true)[:,i]))
    end

    try
        optimize!(mdl,noise=true)
    catch
        ess(mdl;nIter=n_mcmc)
    end

    function objective(theta)
        mdl_out = predict_f(mdl,permutedims(theta'))
        upper_CI = mdl_out[1][1] + 1.645 * mdl_out[2][1]
        return -upper_CI
    end

    temp_ent = Vector{Float64}(undef,n_rep)

    for i in 1:n_opt
        try
            println("Model variance = ",mdl.kernel.σ2)
            println("Model noise = ",mdl.noise)
        catch
        end
        opt_res = bboptimize(
            objective;
            SearchRange = opt_bounds,
            PopulationSize=pop_size,
            MaxTime=max_time,
        )
        norm_vals = best_candidate(opt_res)
        println("Normalized optimum = [$(norm_vals[1]),$(norm_vals[2])]")
        scaled_vals = norm_vals .* (upper_bounds .- lower_bounds) .+ lower_bounds

        if length(lower_bounds) == 2
            temp_design = StepStressTest(
                scaled_vals[1],
                scaled_vals[2],
                design.n_const
            )
        else
            temp_design = StepStressTest(
                scaled_vals[1],
                scaled_vals[2],
                scaled_vals[3]
            )
        end

        @showprogress "Solving initial design..." for j in 1:n_rep
            ent_val = eval_entropy(
                temp_design,
                data,
                samples,
                splines.params.design,
                n_outer,
                n_inner,
                results=:scalar
            )

            temp_ent[j] = ent_val
        end

        for j in 1:n_use
            println("Optim point new entropy = $(temp_ent[j])")
            append!(mdl,permutedims(norm_vals'),[sort(temp_ent,rev=true)[j]])
        end

        #append!(mdl,permutedims(norm_vals'),[mdl_eval])
        println("Appended data = [$(mdl.x[1,end]),$(mdl.x[2,end])]")

        #ess(mdl;nIter=n_mcmc,noise=true)
    end

    _,best_idx = findmax(mdl.y)
    best_inp = vec(mdl.x[:,best_idx]) .* (upper_bounds .- lower_bounds) .+ lower_bounds
    if length(lower_bounds) == 2
        best_design = StepStressTest(
            best_inp[1],
            best_inp[2],
            design.n_const
        )
    else
        best_design = StepStressTest(
            best_inp[1],
            best_inp[2],
            best_inp[3]
        )
    end
    return best_design
end