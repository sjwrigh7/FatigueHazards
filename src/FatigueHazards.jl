module FatigueHazards

using Distributions
using LinearAlgebra
using LatinHypercubeSampling
using ProgressMeter
using GaussianProcesses
using BlackBoxOptim
#using Optim, ADTypes
#import ForwardDiff
using StatsBase
using Plots
# Write your package code here.

include("fatigue_models.jl")
include("data_synthesis.jl")
include("splines.jl")
include("hazard_model.jl")
include("posterior_samplers.jl")
include("mcmc.jl")
include("entropy.jl")
include("prelude.jl")
include("optimization.jl")
end