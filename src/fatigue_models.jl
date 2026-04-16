
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