#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########################### Define Step Size Functions ############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
"""
    stepsize_adjust(diff::Float64,scale::Float64,shape::Float64,offset::Float64)
Function to adjust a given stepsize based on the difference between its calculated acceptance rate and the target.

---
Positional arguments
* `diff::Float64` The difference between the calculated acceptance ratio and the target.
* `scale::Float64` Scaling factor for the equation.
* `shape::Float64` Shape parameter for the equation.
* `offset::Float64` Offset parameter for skewing the adjustment to correct greater in the positive or negative direction

---
Returns
* `factor::Float64` The adjustment by which to multiply the current step size.

---
Details
The function uses an arc-tangent-based form for calculating the adjustment.
The arc-tangent is calculated for the product of the difference and the `shape`.
The arc-tangent is them multiplied by `scale`
This value is then shifted based on the `offset`
* positive values for `offset` correspond to greater corrections of negative differences (the acceptance ratio is too small)
* negative values for `offset` correspond to greater corrections of positive differences (the acceptance ratio is too large)
The exponentiation of this value is returned
"""
function stepsize_adjust(diff::Float64,scale::Float64,shape::Float64,offset::Float64)
    logscale = scale * atan(shape*diff)
    logscale = logscale + offset*sign(logscale) - offset
    factor = exp(logscale)
    return factor
end

"""
    update_stepsize!(acceptance::Vector{Float64},stepsize::StepSize,history::Array{Float64},
    nx::Int,ntheta::Int,target::Vector{Float64},scale::Float64,shape::Float64,offset::Float64,
    weight::Float64)
Function to update the Metropolis-Hastings algorithm step sizes based on the calculated acceptance ratios compared to their target values.

---
Positional arguments
* `acceptance::Vector{Float64}` Array containing the acceptance value for each M-H update.
* `stepsize::StepSize` Struct containing the step sizes used during the M-H updates for θ and ρ.
* `history::Array{Float64}` History of previous step sizes.
* `ntheta::Int` The number of θ variables using M-H updates.
* `nx::Int` The number of ρ variables using M-H updates.
* `target::Vector{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
* `scale::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
* `shape::Float64` Shape parameter to pass into the `stepsize_adjust` function.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
* `weight::Float64` weight to use for the weighted average.
"""
function update_stepsize!(acceptance::Vector{Float64},stepsize::StepSize,
    history::Array{Float64},splines::Splines,target::Vector{Float64},
    scale::Float64,shape::Float64,offset::Float64,weight::Float64)

    @inbounds for i in 1:splines.params.num_basis
        rate = acceptance[i]

        diff = rate - target[1]

        adjustment = stepsize_adjust(diff,scale,shape,offset)
        stepsize.gamma[i] = weighted_avg(
            history[1:(end-1),i],
            history[end,i],
            weight
        )
        stepsize.gamma[i] = weighted_avg(
            repeat(
                [stepsize.gamma[i]],
                size(history)[1]
            ),
            stepsize.gamma[i]*adjustment,
            weight
        )

    end

    rate = acceptance[end]

    diff = rate - target[2]

    adjustment = stepsize_adjust(diff,scale,shape,offset)
    stepsize.beta = weighted_avg(
        history[1:(end-1),end],
        history[end,end],
        weight
    )
    stepsize.beta = weighted_avg(
        repeat(
            [stepsize.beta],
            size(history)[1]
        ),
        stepsize.beta*adjustment,
        weight
    )

    return stepsize
end

"""
    update_stepsize!(acceptance::Vector{Float64},stepsize::StepSize,history::Array{Float64},
    nx::Int,ntheta::Int,target::Vector{Float64},scale::Float64,shape::Float64,offset::Float64,
    weight::Float64)
Function to update the Metropolis-Hastings algorithm step sizes based on the calculated acceptance ratios compared to their target values.

---
Positional arguments
* `acceptance::Vector{Float64}` Array containing the acceptance value for each M-H update.
* `stepsize::StepSize` Struct containing the step sizes used during the M-H updates for θ and ρ.
* `history::Array{Float64}` History of previous step sizes.
* `ntheta::Int` The number of θ variables using M-H updates.
* `nx::Int` The number of ρ variables using M-H updates.
* `target::Vector{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
* `scale::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
* `shape::Float64` Shape parameter to pass into the `stepsize_adjust` function.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
* `weight::Float64` weight to use for the weighted average.
"""
function update_stepsize!(acceptance::Vector{Float64},stepsize::StepSize,
    history::Array{Float64},base_haz_splines::Splines,risk_splines::Splines,
    target::Vector{Float64},scale::Float64,shape::Float64,offset::Float64,weight::Float64)

    n_base = base_haz_splines.params.num_basis
    for i in 1:n_base
        rate = acceptance[i]

        diff = rate - target[1]

        adjustment = stepsize_adjust(diff,scale,shape,offset)
        stepsize.gamma[i] = weighted_avg(
            history[1:(end-1),i],
            history[end,i],
            weight
        )
        stepsize.gamma[i] = weighted_avg(
            repeat(
                [stepsize.gamma[i]],
                size(history)[1]
            ),
            stepsize.gamma[i]*adjustment,
            weight
        )

    end

    for i in 1:risk_splines.params.num_basis
        rate = acceptance[i + n_base]

        diff = rate - target[2]

        adjustment = stepsize_adjust(diff,scale,shape,offset)
        stepsize.beta[i] = weighted_avg(
            history[1:(end-1),i + n_base],
            history[end,i + n_base],
            weight
        )
        stepsize.beta[i] = weighted_avg(
            repeat(
                [stepsize.beta[i]],
                size(history)[1]
            ),
            stepsize.beta[i]*adjustment,
            weight
        )

    end

    return stepsize
end

"""
    weighted_avg(values::Vector{Float64},new::Float64,weight::Float64)
Function to calculate the weighted average of a set of data.
Implementation for weighted average of step size values.
This function assumes a weight of unity for the old values in `values` and uses `weight` for `new`

---
Positional arguments
* `values::Vector{Float64}` Old values for the average.
* `new::Float64` New value added to the average.
* `weight::Float64` weight for the new value.
"""
function weighted_avg(values::Vector{Float64},new::Float64,weight::Float64)
    n = sum(values) + weight*new
    d = length(values) + weight
    avg = n/d
    return avg
end

"""
    weighted_avg(Values::Vector{Bool},new::Vector{Bool},weight::Float64)
Function to calculate the weighted average of a set of data.
Implementation for weighted average of acceptance values.
This function assumes a weight of unity for the old values in `values` and uses `weight` for `new`

---
Positional arguments
* `values::Vector{Bool}` Old acceptance values.
* `new::Vector{Bool}` New acceptance values.
* `weight::Float64` weight for the new values.
"""
function weighted_avg(values::Vector{Bool},new::Vector{Bool},weight::Float64)
    n = sum(values) + weight*sum(new)
    d = length(values) + weight*length(new)
    avg = n/d
    return avg
end

"""
    auto_stepsize(model,data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,
    nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64},
    init::Float64,target::Tuple{Float64},eta::Float64,factor::Float64,offset::Float64)
Function to calculate the appropriate step size for the Metropolis-Hastings algorithm, for a target acceptance ratio.

---
Positional arguments
* `model` Surrogate model.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nbatch::Int` The number of batches of MCMC simulations to run.
* `batchsize::Int` The number of MCMC iterations ro run per batch.
* `prior_data::PriorData` Struct containing information on the variables' prior distributions.
* `nx::Int` The number of x dimensions.
* `ntheta::Int` The number of θ dimensions.
* `nloc::Int` The number of unique settings of x in the experimental data.
* `theta_init::Vector{Float64}` The settings of θ at which to initialize the MCMC.
* `init::Float64` The inital step size for the start of this algorithm.
* `target::Tuple{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
* `scale::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
* `shape::Float64` Shape parameter to pass into the `stepsize_adjust` function.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
---
Returns
* `stepsize::StepSize` Struct containing the calculated stepsizes that will result in the target acceptance rate.
* `stepsize_hist::Array{Float64}` An Array containing the historic data of the stepsizes used in this algorithm.
* `acceptance_hist::Array{Float64}` An Array containing the historic data of the acceptance rates used in this algorithm.

---
Details
This algorithm runs `nbatch` batches of MCMC with M-H updates, each batch having a length of `batchsize`.
The algorithm starts at a step size specified by `init` for the first batch.
After each batch, the algorithm calculates the average acceptance rate of all M-H samples from all preceding batches.
This value is passed to `update_stepsize` to calculate the proposed adjustment to the step size for the next batch.
The step size for the next batch is then calculated as the average of all previous step sizes and the value calculated from `update_stepsize`.
"""
function auto_stepsize(data::StepStressData,splines::Splines,nbatch::Int,batchsize::Int,
    init_vals::Vector{Float64},init::Float64,target::Vector{Float64},
    scale::Float64,shape::Float64,offset::Float64)

    stepsize = StepSize(init,repeat([init],splines.params.num_basis))
    
    total_length = batchsize*nbatch

    stepsize_hist = Array{Float64}(undef,nbatch,splines.params.num_basis+1)
    acceptance_hist = Array{Float64}(undef,nbatch,splines.params.num_basis+1)


    beta_main = Vector{Float64}(undef,total_length)
    gamma_main = Array{Float64}(undef,total_length,splines.params.num_basis)
    main_risk = Vector{Float64}(undef,length(data.t_norm)-2)
    off_risk = Vector{Float64}(undef,length(data.t_norm)-2)
    beta_main_acc = Vector{Bool}(undef,total_length)
    gamma_main_acc = Array{Bool}(undef,total_length,splines.params.num_basis)

    beta_iter = Vector{Float64}(undef,batchsize)
    gamma_iter = Array{Float64}(undef,batchsize,splines.params.num_basis)
    beta_iter_acc = Vector{Bool}(undef,batchsize)
    gamma_iter_acc = Array{Bool}(undef,batchsize,splines.params.num_basis)


    @inbounds @showprogress 1 "Computing Stepsize..." for i in 1:nbatch
        weight = sqrt(i)
        start = (i-1)*batchsize + 1 #+ ifelse(i==1,1,0)
        stop = i*batchsize

        mcmc_baseline_splines!(
            beta_iter,
            gamma_iter,
            main_risk,
            off_risk,
            beta_iter_acc,
            gamma_iter_acc,
            data,
            splines,
            batchsize,
            stepsize,
            init_vals
        )
        
        beta_main[start:stop] .= beta_iter
        gamma_main[start:stop,:] .= gamma_iter
        beta_main_acc[start:stop] .= beta_iter_acc
        gamma_main_acc[start:stop,:] .= gamma_iter_acc

        @inbounds for j in 1:splines.params.num_basis
            stepsize_hist[i,j] = stepsize.gamma[j]
            acceptance_hist[i,j] = weighted_avg(
                gamma_main_acc[1:(start-1),j],
                gamma_main_acc[start:stop,j],
                weight
            )
        end

        stepsize_hist[i,end] = stepsize.beta
        acceptance_hist[i,end] = weighted_avg(
            beta_main_acc[1:(start-1),end],
            beta_main_acc[start:stop,end],
            weight
        )

        stepsize = update_stepsize!(acceptance_hist[i,:],stepsize,
            stepsize_hist[1:i,:],splines,target,scale,shape,offset,
            weight)
    end
    return stepsize,stepsize_hist,acceptance_hist
end

"""
    auto_stepsize(model,data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,
    nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64},
    init::Float64,target::Tuple{Float64},eta::Float64,factor::Float64,offset::Float64)
Function to calculate the appropriate step size for the Metropolis-Hastings algorithm, for a target acceptance ratio.

---
Positional arguments
* `model` Surrogate model.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nbatch::Int` The number of batches of MCMC simulations to run.
* `batchsize::Int` The number of MCMC iterations ro run per batch.
* `prior_data::PriorData` Struct containing information on the variables' prior distributions.
* `nx::Int` The number of x dimensions.
* `ntheta::Int` The number of θ dimensions.
* `nloc::Int` The number of unique settings of x in the experimental data.
* `theta_init::Vector{Float64}` The settings of θ at which to initialize the MCMC.
* `init::Float64` The inital step size for the start of this algorithm.
* `target::Tuple{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
* `scale::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
* `shape::Float64` Shape parameter to pass into the `stepsize_adjust` function.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
---
Returns
* `stepsize::StepSize` Struct containing the calculated stepsizes that will result in the target acceptance rate.
* `stepsize_hist::Array{Float64}` An Array containing the historic data of the stepsizes used in this algorithm.
* `acceptance_hist::Array{Float64}` An Array containing the historic data of the acceptance rates used in this algorithm.

---
Details
This algorithm runs `nbatch` batches of MCMC with M-H updates, each batch having a length of `batchsize`.
The algorithm starts at a step size specified by `init` for the first batch.
After each batch, the algorithm calculates the average acceptance rate of all M-H samples from all preceding batches.
This value is passed to `update_stepsize` to calculate the proposed adjustment to the step size for the next batch.
The step size for the next batch is then calculated as the average of all previous step sizes and the value calculated from `update_stepsize`.
"""
function auto_stepsize(data::StepStressData,base_haz_splines::Splines,risk_splines::Splines,nbatch::Int,
    batchsize::Int,init_vals::Vector{Float64},init::Float64,target::Vector{Float64},
    scale::Float64,shape::Float64,offset::Float64,s_map::Array{Int,2})

    n_base = base_haz_splines.params.num_basis
    n_risk = risk_splines.params.num_basis
    stepsize = StepSize(
        repeat([init],n_risk),
        repeat([init],n_base)
    )
    
    total_length = batchsize*nbatch

    stepsize_hist = Array{Float64}(undef,nbatch,n_risk + n_base)
    acceptance_hist = Array{Float64}(undef,nbatch,n_risk + n_base)


    beta_main = Array{Float64}(undef,total_length,n_risk)
    gamma_main = Array{Float64}(undef,total_length,n_base)
    main_risk = Vector{Float64}(undef,length(data.t_norm)-2)
    off_risk = Vector{Float64}(undef,length(data.t_norm)-2)
    beta_main_acc = Array{Bool}(undef,total_length,n_risk)
    gamma_main_acc = Array{Bool}(undef,total_length,n_base)

    beta_iter = Array{Float64}(undef,batchsize,n_risk)
    gamma_iter = Array{Float64}(undef,batchsize,n_base)
    beta_iter_acc = Array{Bool}(undef,batchsize,n_risk)
    gamma_iter_acc = Array{Bool}(undef,batchsize,n_base)


    @inbounds @showprogress 1 "Computing Stepsize..." for i in 1:nbatch
        weight = sqrt(i)
        start = (i-1)*batchsize + 1 #+ ifelse(i==1,1,0)
        stop = i*batchsize

        mcmc_risk_splines!(
            beta_iter,
            gamma_iter,
            main_risk,
            off_risk,
            beta_iter_acc,
            gamma_iter_acc,
            data,
            base_haz_splines,
            risk_splines,
            batchsize,
            stepsize,
            s_map,
            init_vals
        )
        
        beta_main[start:stop,:] .= beta_iter
        gamma_main[start:stop,:] .= gamma_iter
        beta_main_acc[start:stop,:] .= beta_iter_acc
        gamma_main_acc[start:stop,:] .= gamma_iter_acc

        for j in 1:n_base
            stepsize_hist[i,j] = stepsize.gamma[j]
            acceptance_hist[i,j] = weighted_avg(
                gamma_main_acc[1:(start-1),j],
                gamma_main_acc[start:stop,j],
                weight
            )
        end

        for j in 1:n_risk
            stepsize_hist[i,j + n_base] = stepsize.beta[j]
            acceptance_hist[i,j + n_base] = weighted_avg(
                beta_main_acc[1:(start-1),j],
                beta_main_acc[start:stop,j],
                weight
            )
        end

        stepsize = update_stepsize!(acceptance_hist[i,:],stepsize,
            stepsize_hist[1:i,:],base_haz_splines,risk_splines,
            target,scale,shape,offset,weight)
    end
    return stepsize,stepsize_hist,acceptance_hist
end

"""
    plot_stepsize_opt(stepsize::Array{Float64},acceptance::Array{Float64},nx::Int,
    ntheta::Int,show_plots::Bool,save_plots::Bool,mdl_apnd::String)
Function to plot the results of the M-H stepsize optimization algorithm.

---
Positional arguments
* `stepsize::Array{Float64}` A n by `nx`+`ntheta` Array containing the M-H step sizes used in the stepsize optimization algorithm.
* `acceptance::Array{Float64}` A n by `nx`+`ntheta` Array containing the calculated M-H acceptance rates corresponding to `stepsize`.
* `nx::Int` The number of x dimesions.
* `ntheta::Int` The number of θ dimensions.
* `show_plots::Bool` An indication of whether the plots should be saved.
* `save_plots::Bool` An indication of whether the plots should be displayed.
* `mdl_apnd::String` String to append to the front of the plots' file names.
"""
function plot_stepsize_opt(stepsize::Array{Float64},acceptance::Array{Float64},
            splines::Splines,show_plots::Bool,save_plots::Bool,mdl_apnd::String)
    function plot_stepsize(epochs::Int,stepsize::Vector{Float64},var::String,iter::Int)
        p = Plots.plot(1:epochs,stepsize,label=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
            titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize over Epochs"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-stepsize.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    function plot_acceptance(epochs::Int,acceptance::Vector{Float64},var::String,iter::Int)
        p = Plots.plot(1:epochs,acceptance,label=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance over Epochs"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-acceptance.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    function plot_step_acc(epochs::Int,acceptance::Vector{Float64},stepsize::Vector{Float64},
        var::String,iter::Int)
        epochs = collect(1:epochs)
        p = Plots.scatter(stepsize,acceptance,zcolor=epochs,label=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance vs. Stepsize"))
        xlabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize"))
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-acceptance_v_stepsize.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    function plot_secants(epochs::Int,acceptance::Vector{Float64},stepsize::Vector{Float64},
        var::String,iter::Int)
        p = Plots.plot(1:epochs,stepsize,labe=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize Convergence"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize Rate of Change"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-stepsize_convergence.png") : nothing
        show_plots ? Plots.display(p) : nothing

        p = Plots.plot(1:epochs,acceptance,label=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance Convergence"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance Rate of Change"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-acceptance_convergence.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    secants = assess_convergence(acceptance,stepsize,splines)
    epochs = size(acceptance)[1]
    for gamma in 1:splines.params.num_basis
        plot_stepsize(epochs,stepsize[:,gamma],"gamma",gamma)
        plot_acceptance(epochs,acceptance[:,gamma],"gamma",gamma)
        plot_step_acc(epochs,acceptance[:,gamma],stepsize[:,gamma],"gamma",gamma)
        plot_secants(epochs-1,secants[1][:,gamma],secants[2][:,gamma],"gamma",gamma)
    end

    
    plot_stepsize(epochs,stepsize[:,end],"beta",1)
    plot_acceptance(epochs,acceptance[:,end],"beta",1)
    plot_step_acc(epochs,acceptance[:,end],stepsize[:,end],"beta",1)
    plot_secants(epochs-1,secants[1][:,end],secants[2][:,end],"beta",1)
end

"""
    plot_stepsize_opt(stepsize::Array{Float64},acceptance::Array{Float64},nx::Int,
    ntheta::Int,show_plots::Bool,save_plots::Bool,mdl_apnd::String)
Function to plot the results of the M-H stepsize optimization algorithm.

---
Positional arguments
* `stepsize::Array{Float64}` A n by `nx`+`ntheta` Array containing the M-H step sizes used in the stepsize optimization algorithm.
* `acceptance::Array{Float64}` A n by `nx`+`ntheta` Array containing the calculated M-H acceptance rates corresponding to `stepsize`.
* `nx::Int` The number of x dimesions.
* `ntheta::Int` The number of θ dimensions.
* `show_plots::Bool` An indication of whether the plots should be saved.
* `save_plots::Bool` An indication of whether the plots should be displayed.
* `mdl_apnd::String` String to append to the front of the plots' file names.
"""
function plot_stepsize_opt(stepsize::Array{Float64},acceptance::Array{Float64},
            base_haz_splines::Splines,risk_splines::Splines,show_plots::Bool,
            save_plots::Bool,mdl_apnd::String)
    function plot_stepsize(epochs::Int,stepsize::Vector{Float64},var::String,iter::Int)
        p = Plots.plot(1:epochs,stepsize,label=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
            titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize over Epochs"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-stepsize.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    function plot_acceptance(epochs::Int,acceptance::Vector{Float64},var::String,iter::Int)
        p = Plots.plot(1:epochs,acceptance,label=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance over Epochs"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-acceptance.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    function plot_step_acc(epochs::Int,acceptance::Vector{Float64},stepsize::Vector{Float64},
        var::String,iter::Int)
        epochs = collect(1:epochs)
        p = Plots.scatter(stepsize,acceptance,zcolor=epochs,label=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance vs. Stepsize"))
        xlabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize"))
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-acceptance_v_stepsize.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    function plot_secants(epochs::Int,acceptance::Vector{Float64},stepsize::Vector{Float64},
        var::String,iter::Int)
        p = Plots.plot(1:epochs,stepsize,labe=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize Convergence"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize Rate of Change"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-stepsize_convergence.png") : nothing
        show_plots ? Plots.display(p) : nothing

        p = Plots.plot(1:epochs,acceptance,label=false,top_margin=5Plots.mm,left_margin=5Plots.mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance Convergence"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance Rate of Change"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-acceptance_convergence.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    secants = assess_convergence(acceptance,stepsize,base_haz_splines,risk_splines)
    epochs = size(acceptance)[1]
    n_base = base_haz_splines.params.num_basis
    for gamma in 1:n_base
        plot_stepsize(epochs,stepsize[:,gamma],"gamma",gamma)
        plot_acceptance(epochs,acceptance[:,gamma],"gamma",gamma)
        plot_step_acc(epochs,acceptance[:,gamma],stepsize[:,gamma],"gamma",gamma)
        plot_secants(epochs-1,secants[1][:,gamma],secants[2][:,gamma],"gamma",gamma)
    end

    for beta in 1:risk_splines.params.num_basis
        plot_stepsize(epochs,stepsize[:,beta + n_base],"beta",beta)
        plot_acceptance(epochs,acceptance[:,beta + n_base],"beta",beta)
        plot_step_acc(epochs,acceptance[:,beta + n_base],stepsize[:,beta + n_base],"beta",beta)
        plot_secants(epochs-1,secants[1][:,beta + n_base],secants[2][:,beta + n_base],"beta",beta)
    end
end

"""
    assess_convergence(stepsize::Array{Float64},acceptance::Array{Float64},nx::Int,ntheta::Int)
Function to assess the convergence of the M-H stepsize optimization results.

---
Positional arguments
* `stepsize::Array{Float64}` A n by `nx`+`ntheta` Array containing the M-H step sizes used in the stepsize optimization algorithm.
* `acceptance::Array{Float64}` A n by `nx`+`ntheta` Array containing the calculated M-H acceptance rates corresponding to `stepsize`.
* `nx::Int` The number of x dimesions.
* `ntheta::Int` The number of θ dimensions.

---
Returns
* `secants::Tuple{Array{Float64}}` A Tuple of length 2, with each element being a (n-1) by `nx`+`ntheta` Array.
  * The Arrays contain the calculated secants of the acceptance rates and stepsizes, respectively, calculated as a function of the epochs, relative to the final epoch.

---
Details
This function calculates the slope of the secant line between each element of the `acceptance` and `stepsize` Arrays relative to the last element.
"""
function assess_convergence(stepsize::Array{Float64},acceptance::Array{Float64},
            splines::Splines)

    epochs = size(acceptance)[1]
    breaks = collect(1:epochs)
    temp = Vector{Float64}(undef,length(breaks)-1)
    secants_acceptance = Array{Float64}(undef,length(temp),(splines.params.num_basis + 1))
    secants_stepsize = similar(secants_acceptance)

    function calc_secant!(breaks::Vector{Int},y::Vector{Float64},temp::Vector{Float64})
        for i in eachindex(temp)
            dy = y[end] - y[breaks[i]]
            dx = length(temp) - 1
            temp[i] = dy/dx
        end
    end

    for gamma in 1:splines.params.num_basis
        calc_secant!(breaks,acceptance[:,gamma],temp)
        secants_acceptance[:,gamma] .= temp
        calc_secant!(breaks,stepsize[:,gamma],temp)
        secants_stepsize[:,gamma] .= temp
    end

    calc_secant!(breaks,acceptance[:,end],temp)
    secants_acceptance[:,end] .= temp
    calc_secant!(breaks,stepsize[:,end],temp)
    secants_stepsize[:,end] .= temp
    return secants_acceptance,secants_stepsize
end

"""
    assess_convergence(stepsize::Array{Float64},acceptance::Array{Float64},nx::Int,ntheta::Int)
Function to assess the convergence of the M-H stepsize optimization results.

---
Positional arguments
* `stepsize::Array{Float64}` A n by `nx`+`ntheta` Array containing the M-H step sizes used in the stepsize optimization algorithm.
* `acceptance::Array{Float64}` A n by `nx`+`ntheta` Array containing the calculated M-H acceptance rates corresponding to `stepsize`.
* `nx::Int` The number of x dimesions.
* `ntheta::Int` The number of θ dimensions.

---
Returns
* `secants::Tuple{Array{Float64}}` A Tuple of length 2, with each element being a (n-1) by `nx`+`ntheta` Array.
  * The Arrays contain the calculated secants of the acceptance rates and stepsizes, respectively, calculated as a function of the epochs, relative to the final epoch.

---
Details
This function calculates the slope of the secant line between each element of the `acceptance` and `stepsize` Arrays relative to the last element.
"""
function assess_convergence(stepsize::Array{Float64},acceptance::Array{Float64},
            base_haz_splines::Splines,risk_splines::Splines)

    epochs = size(acceptance)[1]
    breaks = collect(1:epochs)
    temp = Vector{Float64}(undef,length(breaks)-1)
    n_base = base_haz_splines.params.num_basis
    n_risk = risk_splines.params.num_basis
    secants_acceptance = Array{Float64}(undef,length(temp),(n_base + n_risk))
    secants_stepsize = similar(secants_acceptance)

    function calc_secant!(breaks::Vector{Int},y::Vector{Float64},temp::Vector{Float64})
        for i in eachindex(temp)
            dy = y[end] - y[breaks[i]]
            dx = length(temp) - 1
            temp[i] = dy/dx
        end
    end

    for gamma in 1:n_base
        calc_secant!(breaks,acceptance[:,gamma],temp)
        secants_acceptance[:,gamma] .= temp
        calc_secant!(breaks,stepsize[:,gamma],temp)
        secants_stepsize[:,gamma] .= temp
    end

    for beta in 1:n_risk
        calc_secant!(breaks,acceptance[:,beta + n_base],temp)
        secants_acceptance[:,beta + n_base] .= temp
        calc_secant!(breaks,stepsize[:,beta + n_base],temp)
        secants_stepsize[:,beta + n_base] .= temp
    end
    return secants_acceptance,secants_stepsize
end

"""
    find_stepsize(model,data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,
    nx::Int,ntheta::Int,nloc::Int)
    find_stepsize(model,data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,
    nx::Int,ntheta::Int,nloc::Int;theta_init::Vector{Float64},method::Int,make_plots::Bool,
    show_plots::Bool,save_plots::Bool,init::Float64,target::Tuple{Float64},scale::Float64,
    shape::Float64,offset::Float64,mdl_apnd::String)
Function to solve for the appropriate step size for θ and ρ in the Metropolis-Hastings algorithm for Bayesian calibration.

---
Keyword arguments
* `model` Surrogate model
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nbatch::Int` The number of batches of MCMC simulations to run.
* `batchsize::Int` The number of MCMC iterations ro run per batch.
* `prior_data` Struct containing parameter prior distribution informaiton.
* `nx::Int` The number of x dimensions.
* `ntheta::Int` The number of θ dimensions.
* `nloc::Int` The number of unique settings of x in the experimental data.

Keyword arguments
* `theta_init::Union{Vector{Float64},Float64}` The settings of θ at which to initialize the MCMC.
  * default value of 0.5 is used for each dimension of theta if not specified.
* `make_plots::Bool` Indicator of whether to generate plots from the results of the algorithm.
  * default value of true
* `show_plots::Bool` Indicator of whether to display the plots generated.
  * default value of true
* `save_plots::Bool` Indicator of whether to save the plots generated.
  * default value of true
* `init::Float64` The inital step size for the start of this algorithm.
  * By default, 1E-3 is used.
* `target::Vector{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
  * By defualt, 0.3 is used for both.
* `scale::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
  * By default, 2.0 is used.
* `shape::Float64` Shape parameter to pass into the `stepsize_adjust` function.
  * By defualt, 10.0 is used.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
  * By default, 1.5 is used.
* `mdl_apnd::String` String to append to the beginning of generated plot file names.
---
Returns
* `stepsize::StepSize` Struct containing the calculated stepsizes that will result in the target acceptance rate.
"""
function find_stepsize(data::StepStressData,splines::Splines,nbatch::Int,batchsize::Int;
            init_vals::Union{Vector{Float64},Float64}=0.5,
            make_plots::Bool=true,show_plots::Bool=true,save_plots::Bool=true,
            init::Float64=1e-3,target::Vector{Float64}=[0.3,0.3],
            scale::Float64 = 2.0,shape::Float64=10.0,offset::Float64=1.5,mdl_apnd::String="")

    #if typeof(theta_init) == Float64
    #    theta_init = repeat([theta_init],ntheta)
    #end

    stepsize,stepsize_hist,acceptance_hist = auto_stepsize(
        data,
        splines,
        nbatch,
        batchsize,
        init_vals,
        init,
        target,
        scale,
        shape,
        offset
    )

    make_plots ? plot_stepsize_opt(stepsize_hist,acceptance_hist,splines,show_plots,save_plots,mdl_apnd) : nothing

    return stepsize
end

"""
    find_stepsize(model,data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,
    nx::Int,ntheta::Int,nloc::Int)
    find_stepsize(model,data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,
    nx::Int,ntheta::Int,nloc::Int;theta_init::Vector{Float64},method::Int,make_plots::Bool,
    show_plots::Bool,save_plots::Bool,init::Float64,target::Tuple{Float64},scale::Float64,
    shape::Float64,offset::Float64,mdl_apnd::String)
Function to solve for the appropriate step size for θ and ρ in the Metropolis-Hastings algorithm for Bayesian calibration.

---
Keyword arguments
* `model` Surrogate model
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nbatch::Int` The number of batches of MCMC simulations to run.
* `batchsize::Int` The number of MCMC iterations ro run per batch.
* `prior_data` Struct containing parameter prior distribution informaiton.
* `nx::Int` The number of x dimensions.
* `ntheta::Int` The number of θ dimensions.
* `nloc::Int` The number of unique settings of x in the experimental data.

Keyword arguments
* `theta_init::Union{Vector{Float64},Float64}` The settings of θ at which to initialize the MCMC.
  * default value of 0.5 is used for each dimension of theta if not specified.
* `make_plots::Bool` Indicator of whether to generate plots from the results of the algorithm.
  * default value of true
* `show_plots::Bool` Indicator of whether to display the plots generated.
  * default value of true
* `save_plots::Bool` Indicator of whether to save the plots generated.
  * default value of true
* `init::Float64` The inital step size for the start of this algorithm.
  * By default, 1E-3 is used.
* `target::Vector{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
  * By defualt, 0.3 is used for both.
* `scale::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
  * By default, 2.0 is used.
* `shape::Float64` Shape parameter to pass into the `stepsize_adjust` function.
  * By defualt, 10.0 is used.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
  * By default, 1.5 is used.
* `mdl_apnd::String` String to append to the beginning of generated plot file names.
---
Returns
* `stepsize::StepSize` Struct containing the calculated stepsizes that will result in the target acceptance rate.
"""
function find_stepsize(data::StepStressData,base_haz_splines::Splines,risk_splines::Splines,nbatch::Int,
            batchsize::Int,s_map::Array{Int,2};init_vals::Union{Vector{Float64},Float64}=0.5,
            make_plots::Bool=true,show_plots::Bool=true,save_plots::Bool=true,
            init::Float64=1e-3,target::Vector{Float64}=[0.3,0.3],
            scale::Float64 = 2.0,shape::Float64=10.0,offset::Float64=1.5,mdl_apnd::String="")

    #if typeof(theta_init) == Float64
    #    theta_init = repeat([theta_init],ntheta)
    #end

    stepsize,stepsize_hist,acceptance_hist = auto_stepsize(
        data,
        base_haz_splines,
        risk_splines,
        nbatch,
        batchsize,
        init_vals,
        init,
        target,
        scale,
        shape,
        offset,
        s_map
    )
    println(size(stepsize.beta))
    println(size(stepsize.gamma))
    println(size(stepsize_hist))
    println(size(acceptance_hist))
    make_plots ? plot_stepsize_opt(stepsize_hist,acceptance_hist,base_haz_splines,risk_splines,show_plots,save_plots,mdl_apnd) : nothing

    return stepsize
end