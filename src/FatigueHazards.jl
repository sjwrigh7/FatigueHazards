module FatigueHazards

using Distributions
using ProgressMeter
using Optim, ADTypes
# Write your package code here.

include("prelude.jl")
include("fatigue_models.jl")
include("data_synthesis.jl")
include("splines.jl")
include("hazard_model.jl")
include("posterior_samplers.jl")
include("mcmc.jl")
include("entropy.jl")

end