
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

function BilinearMaterial(n_intercept::Float64,slope::Float64,s_offset::Float64;s_max=0,n_target=3)
    if s_max == 0
        s_max = 10^((n_target - n_intercept)/slope) + s_offset
    end
    s_min = 10^((7 - n_intercept)/slope) + s_offset
    println(s_max)
    log_n(s) = n_intercept + slope * log(10,s - s_offset)
    return BilinearMaterial(
        s_max,
        s_offset + 1 + sqrt(eps(Float64)),
        10^(7),
        10^(log_n(s_max))
    )
end

function ModifiedBasquin(n_intercept::Float64,slope::Float64,s_offset::Float64;s_max=200.0)
    log_n(s) = n_intercept + slope * (log(10,s - s_offset))
    # at logS' = 0 and losS' = log(s_max)
    N_min = log_n(s_max)
    N_max = n_intercept

    B = (log(10,s_offset + 1) - log(10,s_max)) / (N_max - N_min)
    log_A = log(10,s_offset + 1) - N_max * B
    return ModifiedBasquin(
        10^(log_A),
        B
    )
end

function BaumelSeeger(s_yield,s_ult,e_yield,e_ult,elasticity)
    if s_yield / elasticity <= 0.003
        psi = 1.0
    else
        psi = 1.375 - 125.0 * s_yield / elasticity
    end
    return BaumelSeeger(
        s_yield,
        s_ult,
        e_yield,
        e_ult,
        elasticity,
        psi
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
    if log_s >= material.s_min
        log_ds = log_s - material.s_min

        log_n = material.n_max + material.slope * log_ds
        n = 10 ^ log_n
    else
        #n = 10 ^ material.n_min
        #n = prevfloat(floatmax(Float64))
        n = 10^15
    end
    return n
end

function _eval_sn(material::Basquin,s::Float64)
    stress_ratio = s / material.strength
    exp_ratio = stress_ratio^(1/material.exponent)
    n = exp_ratio * material.ductility / 2
    return n
end

function _eval_sn(material::ModifiedBasquin,s::Float64)
    #stress_ratio = s / material.coeff
    #exp_ratio = stress_ratio^(1/material.exponent)
    #n = exp_ratio
    log_n = material.coeff + (material.slope) * log(10.0,s)
    n = 10 ^ log_n
    return n
end

function _eval_sn(material::ModifiedBilinear,s::Float64)
    if s <= material.s0
        log_n = material.coeff1 + material.slope1 * log(10.0,s)
    else
        log_n = material.coeff2 + material.slope2 * log(10.0,s)
    end
    return 10 ^ log_n
end

function _eval_sn(material::BaumelSeeger,s::Float64)
    #if s > material.s_yield
    #    slope = (material.e_ult - material.e_yield) / 
    #        (material.s_ult - material.s_yield)
    #    e = material.e_yield + slope * (s - material.s_yield)
    #else
    #    e = s / material.elasticity
    #end

    function f_zero(n)
        elastic_strain = material.s_yield / material.elasticity
        elastic_term = 1.5 * elastic_strain * (2*exp(n))^(-0.087)
        plastic_term = 0.59 * material.psi * (exp(n)) ^ (-0.58)
        
        return elastic_term + plastic_term - 0.5 * s/material.elasticity
    end
    n = find_zero(f_zero,10.0)
    return exp(n)
end

function _eval_sn(material::MMPDS,s::Float64)
    if s <= material.s_offset
        return 1e100
    end
    log_s = log(10,s - material.s_offset)
    log_n = material.n_intercept + material.slope * log_s
    n = 10 ^ (log_n)
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

function _eval_damage(
    damage_rule::PalmgrenMiner,
    material::MaterialModel,
    stresses::Vector{Float64},
    cycles::Vector{Float64},
    error_sample::Float64
)
    damage = 0.0
    cycles_remaining = 0
    for (s,n) in zip(stresses,cycles)
        sn = eval_sn(material,s)
        sn *= 10 ^ error_sample
        damage_ratio = n / sn
        cycles_remaining = (1 - damage) * sn
        damage += damage_ratio
    end
    return damage,cycles_remaining
end

function _eval_damage(
    damage_rule::MansonHalford,
    material::MaterialModel,
    stresses::Vector{Float64},
    cycles::Vector{Float64},
    error_sample::Float64
)
    damage = 0.0
    cycles_remaining = 0
    if length(stresses) == 1
        sn = eval_sn(material,stresses[1])
        curr_sn = sn * 10 ^ error_sample
        damage = cycles[1] / curr_sn
        cycles_remaining = curr_sn
    else
        for i in 1:(length(stresses)-1)
            curr_sn = eval_sn(material,stresses[i])
            curr_sn *= 10 ^ error_sample
            next_sn = eval_sn(material,stresses[i+1])
            next_sn *= 10 ^ error_sample
            
            if i == length(stresses)
                alpha = 1.0
            else
                alpha = (curr_sn / next_sn)^(0.4)
            end
            damage_ratio = (cycles[i] / curr_sn)
            damage = (damage)^(alpha) + damage_ratio
        end
        final_sn = eval_sn(material,stresses[end])
        final_sn *= 10 ^ error_sample
        cycles_remaining = (1 - damage) * final_sn
        damage += cycles[end]/final_sn
    end
    return damage,cycles_remaining
end

function _eval_damage(
    damage_rule::KwofieRhabar,
    material::MaterialModel,
    stresses::Vector{Float64},
    cycles::Vector{Float64},
    error_sample::Float64
)
    damage = 0.0
    cycles_remaining = 0.0
    baseline_sn = eval_sn(material,stresses[1])
    baseline_sn *= 10 ^ error_sample
    for (s,n) in zip(stresses,cycles)
        sn = eval_sn(material,s)
        sn *= 10 ^ error_sample
        damage_ratio = n / sn
        modifier = log(sn) / log(baseline_sn)
        cycles_remaining = (1-damage) / modifier * sn
        damage += damage_ratio * modifier
    end
    return damage,cycles_remaining
end

function _eval_damage(
    damage_rule::ModifiedAeran,
    material::MaterialModel,
    stresses::Vector{Float64},
    cycles::Vector{Float64},
    error_sample::Float64
)
    damage = 0.0
    cycles_remaining = 0.0
    sn = eval_sn(material,stresses[1])
    sn *= 10 ^ error_sample
    delta_i = -1.25/(log(sn))
    if length(stresses) == 1
        if cycles[1] < sn
            damage_i = 1 - (1 - cycles[1]/sn)^(delta_i)
            damage = abs(damage_i)
        else
            damage = 1.0
        end
        cycles_remaining = sn
    else
        curr_sn = eval_sn(material,stresses[1])
        curr_sn *= 10 ^ error_sample
        curr_delta = -1.25/log(curr_sn)
        damage_i = 1 - (1 - cycles[1]/sn)^(curr_delta)
        for i in 2:(length(stresses))
            #println("Current stress = ",stresses[i-1])
            #println("Next stress = ",stresses[i])
            #println("Current damage = ",damage_i)
            next_sn = eval_sn(material,stresses[i])
            next_sn *= 10 ^ error_sample
            mu_interaction = (stresses[i-1]/stresses[i])^2
            next_delta = -1.25/log(next_sn)
            
            n_eff_next = (1 - (1 - damage_i)^(mu_interaction/next_delta)) * next_sn
            n_total = n_eff_next + cycles[i]
            current_ratio = n_total/next_sn
            if current_ratio >= 1.0
                return 1.0,(next_sn - n_eff_next)
            end
            damage_next = 1 - (1 - (current_ratio))^(next_delta)
            #println("Next damage = ",damage_next)
            curr_sn = next_sn
            curr_delta = next_delta
            cycles_remaining = (next_sn - n_eff_next)#(1 - damage_i) * next_sn
            damage_i = damage_next
            damage = abs(damage_next)
        end
    end

    println(damage)
    println(cycles_remaining)
    #=
    for i in eachindex(stresses)
        #println(damage_i)
        sn = eval_sn(material,stresses[i])
        sn *= 10 ^ error_sample
        delta_i = -1.25 / log(sn)
        if i == 1
            damage_ratio = min(cycles[i] / sn,1.0 - sqrt(eps(Float64)))
            damage_i = abs(
                1 - (1 - damage_ratio)^delta_i
            )
            damage = damage_i
        else
            mu_interaction = (stresses[i-1] / stresses[i])^2
            n_eff = sn * (
                1 - (1 - damage_i)^(
                    mu_interaction / delta_i
                )
            )
            n_total = n_eff + cycles[i]
            damage_ratio = min(n_total / sn,1.0 - sqrt(eps(Float64)))
            damage_i = abs(
                1 - (1 - damage_ratio)^delta_i
            )
            damage = damage_i
        end
        if damage >= 1.0
            return damage
        end
    end
    =#
    return damage,cycles_remaining
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