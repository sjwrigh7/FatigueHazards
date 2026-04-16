"""
    m_recurse(k::Int,i::Int,t::Vector{Float64},x::Float64)
Function for recursively evaluating a k-th order M-spline at input x.
---
Positional arguments
* `k::Int`- Order of the spline, used for evaluating recursion condition
* `i::Int`- Index of the spline in question, out of n total splines
* `t::Vector{Float64}`- Vector of spline knot points, having length `k` + `n`
* `x::Float64`- dependent variable used for evaluation
---
Returns
* Scalar sum of polynomial terms evaluated at `x`, whose value depends on `k` and `x`
  * `0.0` if `x` is outside of the knot range `[t[i],t[i+k])`
  * `1 / (t[i+1] - t[i])` if inside the knot range and `k` is 1
  * Recursive evaluations a `k-1` if `k > 1`
"""
function m_recurse(k::Int,i::Int,t::Vector{Float64},x::Float64)
    # i = 1, k = 3 =>
    # i = 1, k = 2 | i = 2, k = 2 =>
    # i = 1, k = 1 | i = 2, k = 1 | i = 3, k = 1

    if t[i] > x || (x) >= t[i+k]
        #println("x out of bounds")
        return 0.0
    end
    if k == 1
            return 1 / (t[i+1]-t[i])
    end

    return k * (
        (x - t[i]) * m_recurse(k-1,i,t,x) +
        (t[i + k] - x) * m_recurse(k-1,i+1,t,x)
    ) / ((k - 1) * (t[i+k] - t[i]))
end

"""
    i_spline()
Generate I splines from M spline basis
"""
function i_spline(k::Int,i::Int,t::Vector{Float64},x::Float64)
    j = findfirst(y -> y > x, t) - 1

    if i > j
        return 0.0
    elseif (j - k + 1) > i
        return 1.0
    end
    
    m_sum = 0.0
    for m in i:j
        m_sum += (t[m+k+1] - t[m]) * 
        m_recurse(k+1,m,t,x) / (k+1)
    end
    return m_sum
end

"""
    generate_knots(k::Int,interior_knots::Vector{Float64},x::Vector{Float64})
Function to generate spline knot grid based on spline order and desired interior knot locations.
---
Positional arguments
* `k::Int`- spline order
* `interior_knots::Vector{Float64}`- vector of desired interior knot locations
* `x::Vector{Float64}`- vector of x settings for regression

---
Returns
* `t::Vector{Float64}`- grid of knots
* `n_splines::Int`- number of splines for the knot grid
"""
function generate_knots(k::Int,interior_knots::Vector{Float64},x::Vector{Float64})
    n_splines = length(interior_knots) + k
    num_knots = n_splines + k
    t = Vector{Float64}(undef,num_knots)

    for i in eachindex(interior_knots)
        t[k+i] = interior_knots[i]
    end
    for i in 1:k
        t[i] = x[1]
        t[i+n_splines] = x[end] +  sqrt(eps(Float64))
    end
    return t,n_splines
end

function eval_m_spline(k::Int,n::Int,t::Vector{Float64},x::Vector{Float64})
    m_evals = Array{Float64}(undef,length(x),n)
    for j in 1:n
        for i in eachindex(x)
            m_evals[i,j] = m_recurse(k,j,t,x[i])
        end
    end
    return m_evals
end

function eval_i_spline(k::Int,n::Int,t::Vector{Float64},x::Vector{Float64})
    i_evals = Array{Float64}(undef,length(x),n)
    t = vcat(
        t,
        t[end]
    )
    for j in 1:n
        for i in eachindex(x)
            #println("basis = $j")
            #println("T index = $i")
            i_evals[i,j] = i_spline(k,j,t,x[i])
        end
    end
    return i_evals
end

function init_splines(spline_order::Int,interior_knots::Vector{Float64},x::Vector{Float64})
    design = SplineDesign(
        spline_order,
        interior_knots
    )
    knot_grid,num_basis = generate_knots(spline_order,interior_knots,x)

    params = SplineParams(
        design,
        knot_grid,
        num_basis
    )

    return params
end

function generate_splines(spline_order::Int,interior_knots::Vector{Float64},x::Vector{Float64})
    params = init_splines(spline_order,interior_knots,x)

    M = eval_m_spline(
        params.design.k,
        params.num_basis,
        params.knot_grid,
        x
    )
    I = eval_i_spline(
        params.design.k,
        params.num_basis,
        params.knot_grid,
        x
    )
    I_diff = diff(I,dims=1)

    splines = Splines(
        params,
        I,
        M,
        I_diff
    )

    return splines
end

function update_x!(splines::Splines,new_x)
    M = eval_m_spline(
        splines.params.design.k,
        splines.params.num_basis,
        splines.params.knot_grid,
        new_x
    )
    I = eval_i_spline(
        splines.params.design.k,
        splines.params.num_basis,
        splines.params.knot_grid,
        new_x
    )
    I_diff = diff(I,dims=1)

    splines.M = M
    splines.I = I
    splines.I_diff = I_diff
end