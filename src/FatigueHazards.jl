module FatigueHazards

using Distributions
using LinearAlgebra
using LatinHypercubeSampling
using ProgressMeter
using GaussianProcesses
using BlackBoxOptim
import Optim, ADTypes
import ForwardDiff
using StatsBase
using Plots
using LaTeXStrings
# Write your package code here.

include("structs.jl")
include("fatigue_models.jl")
include("data_synthesis.jl")
include("splines.jl")
include("hazard_model.jl")
include("posterior_samplers.jl")
include("mcmc.jl")
include("entropy.jl")
include("prelude.jl")
include("optimization.jl")
include("stepsize.jl")
end