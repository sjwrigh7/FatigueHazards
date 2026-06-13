function init_opt_design(s_min=93.4,s_max=210.0,ds_min=-100.0,ds_max=100.0,n_min=1e3,n_max=1e7;reduce=false,n_const=5e3)
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

function log_doe(lin_doe_norm,l_bounds,u_bounds)
    log_doe = similar(lin_doe_norm)
    for j in axes(log_doe,2)
        if l_bounds[j] > 0
            log_min = log(l_bounds[j])
            log_max = log(u_bounds[j])
            log_doe[:,j] .= exp.(
                lin_doe_norm[:,j] .* (log_max - log_min) .+ log_min
            )
        elseif l_bounds[j] <= 0 && u_bounds[j] > 0
            log_min = min(4,log(u_bounds[j]) - 4)
            min_abs = log(abs(u_bounds[j]))
            max_abs = log(abs(u_bounds[j]))
            zero_pos = (min_abs - log_min) / (min_abs - log_min + max_abs - log_min)
            curr_vec = copy(lin_doe_norm[:,j])
            for i in eachindex(curr_vec)
                if curr_vec[i] <= zero_pos
                    log_doe[i,j] = -exp(
                        ((zero_pos - curr_vec[i]) / 
                        zero_pos) * (min_abs - log_min) .+
                        log_min
                    )
                else
                    log_doe[i,j] = exp(
                        (curr_vec[i] - zero_pos) / (1 - zero_pos) *
                        (max_abs - log_min) .+ log_min
                    )
                end
            end
        elseif l_bounds[j] < 0 && u_bounds[j] <= 0
            log_min = log(abs(u_bounds[j]))
            log_max = log(abs(l_bounds[j]))
            log_doe[:,j] .= -exp.(
                lin_doe_norm[:,j] .* (log_max - log_min) .+ log_min
            )
        end
    end
    log_doe_norm = (log_doe .- l_bounds') ./ (u_bounds - l_bounds)'
    return log_doe,log_doe_norm
end

function lhc_sampler(plan)
    design = Array{Float64}(undef,size(plan))
    n_breaks = size(plan,1) + 1
    bin_vals = range(
        start = 0.0,
        stop = 1.0,
        length = n_breaks
    )
    @inbounds for j in axes(plan,2)
        @inbounds for i in axes(plan,1)
            bin_num = plan[i,j]
            unif_range = Uniform(bin_vals[bin_num],bin_vals[bin_num + 1])
            rand_val = rand(unif_range)
            design[i,j] = rand_val
        end
    end
    return design
end

function init_data(
    design::OptDesign,
    samples::PosteriorIID,
    data::StepStressData,
    base_haz_splines::Splines,
    constraints::TestConstraints;
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
    doe_init = LHCoptim(n_init,3,20)
    doe_norm = lhc_sampler(doe_init[1])
    doe = scaleLHC(doe_norm,doe_bounds)

    lower_bounds = [p[1] for p in doe_bounds]
    upper_bounds = [p[2] for p in doe_bounds]

    #doe_norm = (doe .- lower_bounds') ./ (upper_bounds' .- lower_bounds')

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
                base_haz_splines,
                n_outer,
                n_inner,
                constraints;
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
    base_haz_splines::Splines,
    constraints::TestConstraints;
    n_init=15,
    n_rep=7,
    n_inner=2500,
    n_outer=2500
)

    doe_bounds = [
        (design.s_min,design.s_max),
        (design.ds_min,design.ds_max),
    ]
    doe_init = LHCoptim(n_init,2,20)
    doe_norm = lhc_sampler(doe_init[1])
    doe = scaleLHC(doe_norm,doe_bounds)

    lower_bounds = [p[1] for p in doe_bounds]
    upper_bounds = [p[2] for p in doe_bounds]

    #doe_norm = (doe .- lower_bounds') ./ (upper_bounds' .- lower_bounds')

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
                base_haz_splines,
                n_outer,
                n_inner,
                constraints;
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
    base_haz_splines::Splines,
    n_opt::Int,
    constraints::TestConstraints;
    s_min=1e3,
    s_max=2e4,
    ds_min=-1e4,
    ds_max=1e4,
    n_min=1e3,
    n_max=1e7,
    n_const=5e3,
    reduce=false,
    n_init=0,
    n_rep=3,
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
    #println(n_init)

    lower_bounds,upper_bounds,doe_norm,doe_resp = init_data(
        design,
        samples,
        data,
        base_haz_splines,
        constraints;
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
        #println("Initial DOE Shannon entropy = ")
        #println(vec(sort(doe_resp,dims=2)[:,i]))
        append!(
            mdl,
            permutedims(doe_norm),
            vec(sort(doe_resp,dims=2,rev=true)[:,i])
        )
    end

    try
        optimize!(mdl,noise=true)
    catch
        ess(mdl;nIter=n_mcmc)
    end

    function objective_max_upper_ci(theta)
        mdl_out = predict_f(mdl,permutedims(theta'))
        upper_CI = mdl_out[1][1] + 1.645 * mdl_out[2][1]
        return -upper_CI
    end

    function objective_max_expected_improvement(theta)
        mdl_out = predict_f(mdl,permutedims(theta'))
        mu = mdl_out[1][1]
        sig = sqrt(mdl_out[2][1])

        ei = (mu - curr_max) * 
            cdf(Normal(mu,sig),curr_max) +
            sig * 
            pdf(Normal(mu,sig),curr_max)
        return -ei
    end

    temp_ent = Vector{Float64}(undef,n_rep)

    @showprogress "Running Bayesian optimization..." for i in 1:n_opt
        curr_max,_ = findmax(mdl.y)
        opt_res = bboptimize(
            objective_max_expected_improvement;
            SearchRange = opt_bounds,
            PopulationSize=pop_size,
            MaxTime=max_time,
            TraceMode = :silent
        )
        norm_vals = best_candidate(opt_res)
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

        for j in 1:n_rep
            ent_val = eval_entropy(
                temp_design,
                data,
                samples,
                base_haz_splines,
                n_outer,
                n_inner,
                constraints;
                results=:scalar
            )

            temp_ent[j] = ent_val
        end

        for j in 1:n_use
            append!(
                mdl,
                permutedims(norm_vals'),
                [sort(temp_ent,rev=true)[j]]
            )
        end

        #append!(mdl,permutedims(norm_vals'),[mdl_eval])
        #println("Appended data = [$(mdl.x[1,end]),$(mdl.x[2,end])]")

        #ess(mdl;nIter=n_mcmc,noise=true)
        try
            optimize!(mdl)
        catch
            ess(mdl,nIter=n_mcmc)
        end
    end

    validation_doe_init = LHCoptim(4,3,20)
    validation_doe_norm = lhc_sampler(validation_doe_init[1])
    validation_resp = Vector{Float64}(undef,size(validation_doe_norm,1))
    validation_prediction = similar(validation_resp)
    validation_uncert = similar(validation_resp)
    for i in axes(validation_doe_norm,1)
        scaled_vals = validation_doe_norm[i,:] .* (upper_bounds .- lower_bounds) .+ lower_bounds

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
        validation_resp[i] = eval_entropy(
            temp_design,
            data,
            samples,
            base_haz_splines,
            n_outer,
            n_inner,
            constraints;
            results=:scalar
        )
        mdl_out = predict_f(mld,permutedims(validation_doe_norm[i,:]'))
        validation_predict[i] = mdl_out[1][1]
        validation_uncert[i] = mdl_out[2][1]
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
    return best_design,mdl.x,mdl.y,validation_doe_norm,validation_resp,validation_prediction,validation_uncert
end

function init_data(
    design::OptDesign,
    samples::PosteriorIID,
    data::StepStressData,
    base_haz_splines::Splines,
    risk_splines::Splines,
    constraints::TestConstraints;
    n_init=30,
    n_rep=3,
    n_inner=2500,
    n_outer=2500,
    obj=:entropy
)

    doe_bounds = [
        (design.s_min,design.s_max),
        (design.ds_min,design.ds_max),
        (design.n_min,design.n_max)
    ]
    doe_init = LHCoptim(n_init,3,20)
    doe_norm = lhc_sampler(doe_init[1])
    #doe = scaleLHC(doe_norm,doe_bounds)

    lower_bounds = [p[1] for p in doe_bounds]
    upper_bounds = [p[2] for p in doe_bounds]

    doe,doe_norm = log_doe(doe_norm,lower_bounds,upper_bounds)
    #doe = doe_norm .* (upper_bounds .- lower_bounds)' .+ lower_bounds'

    #doe_norm = (doe .- lower_bounds') ./ (upper_bounds' .- lower_bounds')

    doe_resp = Array{Float64}(undef,n_init,n_rep)
    #println(doe_norm)
    #println(doe)

    @showprogress "Solving initial design..." for i in 1:n_init
        temp_design = StepStressTest(
            doe[i,1],
            doe[i,2],
            doe[i,3]
        )
        open("../examples/temp_designs.txt","a") do f
            println(f,join(string.(doe[i,:]),','))
        end
        #println("n = $(temp_design.n)")
        for j in 1:n_rep
            ent_val,times = eval_entropy(
                temp_design,
                data,
                samples,
                base_haz_splines,
                risk_splines,
                n_outer,
                n_inner,
                constraints;
                results=:scalar,
                return_times=true
            )

            scaled_times = times .* data.t_max
            if obj == :entropy
                doe_resp[i,j] = ent_val
            elseif obj == :modified
                doe_resp[i,j] = ent_val / log(mean(scaled_times))
            end
        end
    end

    return lower_bounds,upper_bounds,doe_norm,doe_resp
end

function init_data(
    design::OptDesignReduced,
    samples::PosteriorIID,
    data::StepStressData,
    base_haz_splines::Splines,
    risk_splines::Splines,
    constraints::TestConstraints;
    n_init=15,
    n_rep=3,
    n_inner=2500,
    n_outer=2500,
    multithread=true,
    obj=:entropy
)

    doe_bounds = [
        (design.s_min,design.s_max),
        (design.ds_min,design.ds_max),
    ]
    doe_init = LHCoptim(n_init,2,20)
    doe_norm = lhc_sampler(doe_init[1])
    #doe = scaleLHC(doe_norm,doe_bounds)

    lower_bounds = [p[1] for p in doe_bounds]
    upper_bounds = [p[2] for p in doe_bounds]

    doe,doe_norm = log_doe(doe_norm,lower_bounds,upper_bounds)
    #doe_norm = (doe .- lower_bounds') ./ (upper_bounds' .- lower_bounds')

    doe_resp = Array{Float64}(undef,n_init,n_rep)

    @showprogress "Solving initial design..." for i in 1:n_init
        temp_design = StepStressTest(
            doe[i,1],
            doe[i,2],
            design.n_const
        )
        open("../examples/temp_designs.txt","a") do f
            println(f,join(string.(doe[i,:]),','))
        end
        for j in 1:n_rep
            ent_val,times = eval_entropy(
                temp_design,
                data,
                samples,
                base_haz_splines,
                risk_splines,
                n_outer,
                n_inner,
                constraints;
                results=:scalar,
                multithread=multithread,
                return_times=true
            )

            scaled_times = times .* data.t_max
            if obj == :entropy
                doe_resp[i,j] = ent_val
            elseif obj == :modified
                doe_resp[i,j] = ent_val / log(mean(scaled_times))
            end
        end
    end

    return lower_bounds,upper_bounds,doe_norm,doe_resp
end

function optimize_design(
    samples::PosteriorIID,
    data::StepStressData,
    base_haz_splines::Splines,
    risk_splines::Splines,
    n_opt::Int,
    constraints::TestConstraints;
    s_min=93.4,
    s_max=210.0,
    ds_min=-100.0,
    ds_max=100.0,
    n_min=1e3,
    n_max=1e7,
    n_const=5e3,
    reduce=false,
    n_init=0,
    n_rep=3,
    n_use=3,
    n_inner=2500,
    n_outer=2500,
    n_mcmc=20000,
    pop_size=5000,
    max_time=1.5,
    multithread=true,
    obj=:entropy,
    n_validate=0
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
    #println(design)
    lower_bounds,upper_bounds,doe_norm,doe_resp = init_data(
        design,
        samples,
        data,
        base_haz_splines,
        risk_splines,
        constraints;
        n_init=n_init,
        n_rep=n_rep,
        n_inner=n_inner,
        n_outer=n_outer,
        obj=obj
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
        append!(
            mdl,
            permutedims(doe_norm),
            vec(sort(doe_resp,dims=2,rev=true)[:,i])
        )
    end

    try
        optimize!(mdl,noise=true)
    catch
        ess(mdl;nIter=n_mcmc)
    end

    function objective_max_upper_ci(theta)
        mdl_out = predict_f(mdl,permutedims(theta'))
        upper_CI = mdl_out[1][1] + 1.645 * mdl_out[2][1]
        return -upper_CI
    end

    function objective_max_expected_improvement(theta)
        mdl_out = predict_f(mdl,permutedims(theta'))
        mu = mdl_out[1][1]
        sig = sqrt(mdl_out[2][1])

        ei = (mu - curr_max) * 
            cdf(Normal(mu,sig),curr_max) +
            sig * 
            pdf(Normal(mu,sig),curr_max)
        return -ei
    end

    curr_max,_ = findmax(mdl.y)
    temp_ent = Vector{Float64}(undef,n_rep)

    n_optim_evals = 20
    opt_mins = Vector{Float64}(undef,n_optim_evals)
    opt_inps = Array{Float64}(undef,n_optim_evals,length(lower_bounds))


    @showprogress "Running Bayesian optimization..." for i in 1:n_opt
        curr_max = max(curr_max,mdl.y[end])
        #opt_res = bboptimize(
        #    objective_max_expected_improvement;
        #    SearchRange = opt_bounds,
        #    PopulationSize=pop_size,
        #    MaxTime=max_time,
        #    TraceMode = :silent
        #)
        #norm_vals = best_candidate(opt_res)
        x0_doe = LHCoptim(n_optim_evals,length(lower_bounds),5)[1]
        x0 = lhc_sampler(x0_doe)
        #x0 .= x0 .* (upper_bounds .- lower_bounds .- 2 * sqrt(eps(Float64)))' .+ lower_bounds' .+ sqrt(eps(Float64))

        for j in axes(x0,1)
            temp = Optim.optimize(
                objective_max_expected_improvement,
                sqrt(eps(Float64)),
                1 - sqrt(eps(Float64)),
                #lower_bounds,
                #upper_bounds,
                x0[j,:],
                Optim.Fminbox(Optim.LBFGS()),
                autodiff=ADTypes.AutoForwardDiff()
            )
            opt_mins[j] = temp.minimum
            opt_inps[j,:] .= temp.minimizer
        end

        _,min_idx = findmin(opt_mins)
        norm_vals = opt_inps[min_idx,:]
        #println(norm_vals)

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
        open("../examples/temp_designs.txt","a") do f
            println(f,join(string.(scaled_vals),','))
        end

        temp_ent .= -10.0
        for j in 1:n_rep
            ent_val,times = eval_entropy(
                temp_design,
                data,
                samples,
                base_haz_splines,
                risk_splines,
                n_outer,
                n_inner,
                constraints;
                results=:scalar,
                multithread=multithread,
                return_times=true
            )

            scaled_times = times .* data.t_max
            if obj == :entropy
                temp_ent[j] = ent_val
            elseif obj == :modified
                temp_ent[j] = ent_val / log(mean(scaled_times))
            end
        end

        for j in 1:n_use
            #println("Optim point new entropy = $(sort(temp_ent,rev=true)[j])")
            append!(
                mdl,
                permutedims(norm_vals'),
                [sort(temp_ent,rev=true)[j]]
            )
        end

        #append!(mdl,permutedims(norm_vals'),[mdl_eval])
        #println("Appended data = [$(mdl.x[1,end]),$(mdl.x[2,end])]")

        #ess(mdl;nIter=n_mcmc,noise=true)
        try
            optimize!(mdl)
        catch
            ess(mdl,nIter=n_mcmc)
        end
    end

    if n_validate > 0
        validation_doe_init = LHCoptim(n_validate,length(lower_bounds),20)
        validation_doe_norm = lhc_sampler(validation_doe_init[1])
        validation_resp = Vector{Float64}(undef,size(validation_doe_norm,1))
        validation_predict = similar(validation_resp)
        validation_uncert = similar(validation_resp)
        for i in axes(validation_doe_norm,1)
            scaled_vals = validation_doe_norm[i,:] .* (upper_bounds .- lower_bounds) .+ lower_bounds

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
            validation_resp[i] = eval_entropy(
                temp_design,
                data,
                samples,
                base_haz_splines,
                risk_splines,
                n_outer,
                n_inner,
                constraints;
                results=:scalar
            )
            mdl_out = predict_f(mdl,permutedims(validation_doe_norm[i,:]'))
            validation_predict[i] = mdl_out[1][1]
            validation_uncert[i] = mdl_out[2][1]
        end
    else
        validation_doe_norm = nothing
        validation_resp = nothing
        validation_predict = nothing
        validation_uncert = nothing
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
    return best_design,mdl.x,mdl.y,validation_doe_norm,validation_resp,validation_predict,validation_uncert
end