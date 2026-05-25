
function BilinearMaterial(s_max::Real,s_min::Real,n_max::Real,n_min::Real)
    delta_s = log(10,s_max) - log(10,s_min)
    delta_n = log(10,n_min) - log(10,n_max)

    slope = delta_n / delta_s

    return BilinearMaterial(
        log(10,s_max),
        log(10,s_min),
        log(10,n_max),
        log(10,n_min),
        Float64(slope)
    )
end

function bilinear_sn(material::BilinearMaterial,s::Float64)
    log_s = log(10,s)
    if log_s < material.s_max
        log_ds = log_s - material.s_min

        log_n = material.n_max + material.slope * log_ds
        return 10 ^ log_n
    else
        return 10 ^ material.n_min
    end
end

function palmgren_miner(material::BilinearMaterial,stresses::Vector{Float64},cycles::Vector{Float64},error_sample=0.0)
    damage = 0.0
    for (s,n) in zip(stresses,cycles)
        sn = bilinear_sn(material,s)
        sn = sn * 10 ^ error_sample
        ratio = n / sn
        damage += ratio
    end
    return damage        
end

function eval_sn(material::MaterialModel,stress::Float64)
    cycles = _eval_sn(material,stress)
    return cycles
end

function _eval_sn(material::BilinearMaterial,s::Float64)
    log_s = log(10,s)
    if log_s < material.s_max
        log_ds = log_s - material.s_min

        log_n = material.n_max + material.slope * log_ds
        n = 10 ^ log_n
    else
        n = 10 ^ material.n_min
    end
    return n
end

function eval_damage(
    damage_rule::DamageRule,
    material::MaterialModel,
    stresses::Vector{Float64},
    cycles::Vector{Float64},error_sample::Float64)
    
    damage = _eval_damage(
        damage_rule,
        material,
        stresses,
        cycles,
        error_sample
    )
    return damage
end

function _eval_damage(
    damage_rule::LinearDamage,
    material::MaterialModel,
    stresses::Vector{Float64},
    cycles::Vector{Float64},
    error_sample::Float64
    )
    
    damage = 0.0
    for (s,n) in zip(stresses,cycles)
        sn = eval_sn(material,s)
        sn *= 10 ^ error_sample
        base_ratio = n / sn
        s_diff = s - damage_rule.x0
        modifier = damage_rule.intercept + s_diff * damage_rule.coeff
        damage += base_ratio * modifier
    end
    return damage
end

function _eval_damage(
    damage_rule::PolynomialDamage,
    material::MaterialModel,
    stresses::Vector{Float64},
    cycles::Vector{Float64},
    error_sample::Float64
)

    damage = 0.0
    exponents = collect(1.0:1.0:damage_rule.order)
    for (s,n) in zip(stresses,cycles)
        sn = eval_sn(material,s)
        sn *= 10.0 ^ error_sample
        base_ratio = n / sn
        s_diff = (s - damage_rule.x0) .^ exponents
        modifier = damage_rule.intercept + sum(s_diff .* damage_rule.coeff)
        damage += base_ratio * modifier
    end
    return damage
end

function calc_remainder(
    damage_rule::DamageRule,
    material::MaterialModel,
    stress::Float64,
    error_sample::Float64,
    prev_damage::Float64
)
    remaining_cycles = _calc_remainder(
        damage_rule,
        material,
        stress,
        error_sample,
        prev_damage
    )
    return remaining_cycles
end

function _calc_remainder(
    damage_rule::LinearDamage,
    material::MaterialModel,
    stress::Float64,
    error_sample::Float64,
    prev_damage::Float64
)
    remaining_damage = 1.0 - prev_damage

    base_sn = eval_sn(material,stress) * 10.0 ^ error_sample

    s_diff = stress - damage_rule.x0
    modifier = damage_rule.intercept + s_diff * damage_rule.coeff
    cycles_remaining = base_sn * remaining_damage / modifier

    return cycles_remaining
end

function _calc_remainder(
    damage_rule::PolynomialDamage,
    material::MaterialModel,
    stress::Float64,
    error_sample::Float64,
    prev_damage::Float64
)
    remaining_damage = 1.0 - prev_damage
    
    base_sn = eval_sn(material,stress) * 10.0 ^ error_sample

    exponents = collect(1.0:1.0:damage_rule.order)
    s_diff = (stress - damage_rule.x0) .^ exponents
    modifier = damage_rule.intercept + sum(s_diff .* damage_rule.coeff)

    cycles_remaining = base_sn * remaining_damage / modifier

    return cycles_remaining
end