struct StepStressTest
    s0::Float64
    ds::Float64
    n::Float64
end

function single_step_stress(material::BilinearMaterial,test::StepStressTest,error_model::UnivariateDistribution)
    # initialize to not have failed
    has_failed = false
    # incrementor for number of stress steps
    ita = 0

    stresses = []
    cycles = []
    cumulative_damage = Float64[]

    push!(cumulative_damage,0.0)
    # sample material strength variance on the specimen level
    error_sample = rand(error_model)

    while !has_failed
        stresses = test.s0 .+ collect(0:1:ita) * test.ds
        cycles = repeat([test.n],ita + 1)
        
        damage_i = palmgren_miner(material,stresses,cycles,error_sample)

        push!(
            cumulative_damage,
            cumulative_damage[end] + damage_i
        )

        has_failed = cumulative_damage[end] >= 1.0

        ita += 1
    end

    prev_iter_damage = cumulative_damage[end-1]
    remaining_damage = 1 - prev_iter_damage

    sn = bilinear_sn(material,stresses[end]) * 10 ^ error_sample

    cycles[end] = remaining_damage * sn
    println(cycles)
    return stresses,cycles
end

struct StepStressSweep
    s0::Vector{Float64}
    ds::Vector{Float64}
    n::Vector{Float64}
end

function sweep_design(s0::Vector{Float64},ds::Vector{Float64},n::Vector{Float64},n_rep::Int)
    total_count = length(s0) * length(ds) * length(n) * n_rep

    design = Vector{StepStressTest}(undef,total_count)
    #Array{Float64}(undef,total_count,3)

    ita = 0
    for k in n
        for j in ds
            for i in s0
                for l in 1:n_rep
                    ita += 1
                    design[ita] = StepStressTest(
                        i,
                        j,
                        k
                    )
                end
            end
        end
    end

    #return StepStressSweep(
    #    design[:,1],
    #    design[:,2],
    #    design[:,3]
    #)
    return design
end

struct StepStressRawData
    stresses::Vector{Vector{Float64}}
    cycles::Vector{Vector{Float64}}
end

struct StepStressData
    raw::StepStressRawData
    s_max::Float64
    t_max::Float64
    s_norm::Array{Float64,2}
    t_norm::Array{Float64}
    delta_i::Array{Int,2}
end

function simulate_step_stress(material::BilinearMaterial,test::Vector{StepStressTest},error_model::UnivariateDistribution)
    stresses = Vector{Float64}[]
    cycles = Vector{Float64}[]

    for i in eachindex(test)
        s,n = single_step_stress(material,test[i],error_model)
        push!(stresses,s)
        push!(cycles,n)
    end

    raw_data = StepStressRawData(
        stresses,
        cycles
    )

    clean_data = partition_time(raw_data)

    return clean_data
end

function partition_time(data::StepStressRawData)
    time_set = [cumsum(cycle) for cycle in data.cycles]

    time_set = reduce(vcat,time_set)

    time_set = vcat(
        0.0,
        sort(unique(time_set)),
        Inf
    )

    delta_i = Array{Int}(undef,length(time_set),length(data.cycles))
    delta_i[1,:] .= 0
    delta_i[end,:] .= 0

    stresses = Array{Float64}(undef,size(delta_i))
    stresses[1,:] .= 0.0
    stresses[end,:] .= 0.0

    for j in axes(delta_i,2)
        for i in 2:(size(delta_i,1) - 1)
            if data.cycles[j][1] > time_set[i]
                cycle_idx = 1
            elseif sum(data.cycles[j]) < time_set[i]
                cycle_idx = length(data.cycles[j])
            else
                cycle_idx = findlast(x -> x <= time_set[i],cumsum(data.cycles[j]))
            end
            stresses[i,j] = data.stresses[j][cycle_idx]

            if isapprox(sum(data.cycles[j]), time_set[i])
                delta_i[i,j] = 1
            else
                delta_i[i,j] = 0
            end
        end
    end

    s_max = maximum(stresses)
    t_max = time_set[end-1]

    s_norm = stresses ./ s_max
    t_norm = time_set ./ t_max

    println(typeof(delta_i))

    clean_data = StepStressData(
        data,
        s_max,
        t_max,
        s_norm,
        t_norm,
        delta_i
    )
    println(typeof(clean_data.delta_i))
    return clean_data
end

function merge_grids(prev_time,prev_stress,new_time,fail_idx)
    new_stress = Vector{Float64}(undef,length(fail_idx))
    for i in eachindex(new_stress)
        new_stress[i] = prev_stress[
            fail_idx[i] + 1
        ]
    end
    
    merged_time = vcat(
        new_time,
        prev_time
    )

    merged_stress = vcat(
        new_stress,
        prev_stress
    )

    sort_idx = sortperm(merged_time)

    sort_time = merged_time[sort_idx]
    sort_stress = merged_stress[sort_idx]

    time_grid_idx = findall(x -> in(x,1:length(new_time)),sort_idx)
    param_idx = sort_idx[time_grid_idx]

    sort_time,sort_stress,time_grid_idx,param_idx
end