# material models
struct BilinearMaterial
    s_max::Float64
    s_min::Float64
    n_max::Float64
    n_min::Float64
    slope::Float64
end

# data synthesis structs
struct StepStressTest
    s0::Float64
    ds::Float64
    n::Float64
end

struct StepStressSweep
    s0::Vector{Float64}
    ds::Vector{Float64}
    n::Vector{Float64}
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
    in_risk_idx::Vector{Vector{Int}}
end

# mcmc structs
struct PosteriorSamples
    beta::Union{Array{Float64,1},Array{Float64,2}}
    gamma::Array{Float64,2}
    beta_accept::Union{Array{Bool,1},Array{Bool,2}}
    gamma_accept::Array{Bool,2}
end

struct PosteriorIID
    beta::Union{Array{Float64,1},Array{Float64,2}}
    gamma::Array{Float64,2}
end

# optimal design structs
struct OptDesign
    s_min::Float64
    s_max::Float64
    ds_min::Float64
    ds_max::Float64
    n_min::Float64
    n_max::Float64
end

struct OptDesignReduced
    s_min::Float64
    s_max::Float64
    ds_min::Float64
    ds_max::Float64
    n_const::Float64
end

# spline structs
struct SplineDesign
    k::Int
    interior_knots::Vector{Float64}
end

struct SplineParams
    design::SplineDesign
    knot_grid::Vector{Float64}
    num_basis::Int
end

mutable struct Splines
    params::SplineParams
    I::Array{Float64,2}
    M::Array{Float64,2}
    I_diff::Array{Float64,2}
end

# stepsize
mutable struct StepSize
    beta::Union{Vector{Float64},Float64}
    gamma::Vector{Float64}
end
