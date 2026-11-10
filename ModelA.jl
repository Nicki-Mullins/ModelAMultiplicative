# ModelA.jl
#
# Functions to simulate model A with the Metropolis algorithm
# H[φ] = Σ_x [ 1/2 (∇φ)^2 + m²/2 φ² + λ/4 φ⁴ ]
# Nonlinear relaxation rate is used, GammaA = GammaA0 + GammaA2 phi^2

const lambd = 4.0
const GammaA0 = 1.0
const GammaA2 = 0.2

const e_mu = (CartesianIndex(1,0,0), CartesianIndex(0,1,0), CartesianIndex(0,0,1))

function sites_with_parity(parity::Int, Nx::Int, Ny::Int, Nz::Int)
    return [(x, y, z) for x in 0:Nx-1 for y in 0:Ny-1 for z in 0:Nz-1
            if (x + y + z) % 2 == parity]
end

function relaxation(phi::Float64, a::Float64, b::Float64)
    return a + b * phi^2
end

function delta_H(x::Int, y::Int, z::Int, phi::Array{Float64,3},
                 phi_trial::Float64, h::Float64, L::Int,
                 m2::Float64)

    phi_old = phi[x+1, y+1, z+1]
    dphi    = phi_trial - phi_old
    dphi2   = phi_trial^2 - phi_old^2

    xp = phi[mod(x+1,L)+1, y+1,          z+1          ]
    xm = phi[mod(x-1,L)+1, y+1,          z+1          ]
    yp = phi[x+1,          mod(y+1,L)+1,  z+1          ]
    ym = phi[x+1,          mod(y-1,L)+1,  z+1          ]
    zp = phi[x+1,          y+1,           mod(z+1,L)+1 ]
    zm = phi[x+1,          y+1,           mod(z-1,L)+1 ]

    neighbor_sum = xp + xm + yp + ym + zp + zm

    kinetic   = (3 * dphi2 - dphi * neighbor_sum)
    potential = 0.5 * m2 * dphi2 + 0.25 * lambd * (phi_trial^4 - phi_old^4)
    source    = -h * dphi

    return kinetic + potential + source
end

function ModelA_update(phi::Array{Float64,3}, h::Float64, dt::Float64, L::Int,
                           m2::Float64)

    phi_new = copy(phi)

    for parity in [0, 1]
        for (i, j, k) in sites_with_parity(parity, L, L, L)

            GammaA = relaxation(phi_new[i+1, j+1, k+1], GammaA0, GammaA2)

            phi_old   = phi_new[i+1, j+1, k+1]
            phi_trial = phi_old + sqrt(2 * GammaA * dt) * randn()

            dH = delta_H(i, j, k, phi_new, phi_trial, h, L, m2)

            if dH <= 0 || rand() < exp(-dH)
                phi_new[i+1, j+1, k+1] = phi_trial
            end
        end
    end

    return phi_new
end

function ModelA_update_hastings(phi::Array{Float64,3}, h::Float64, dt::Float64, L::Int,
                                m2::Float64)

    phi_new = copy(phi)

    for parity in [0, 1]
        for (i, j, k) in sites_with_parity(parity, L, L, L)

            phi_old   = phi_new[i+1, j+1, k+1]
            Gamma_old = relaxation(phi_old, GammaA0, GammaA2)

            phi_trial = phi_old + sqrt(2 * Gamma_old * dt) * randn()
            Gamma_new = relaxation(phi_trial, GammaA0, GammaA2)

            dH = delta_H(i, j, k, phi_new, phi_trial, h, L, m2)

            # log[ q(trial -> old) / q(old -> trial) ]
            log_q_ratio = 0.5 * log(Gamma_old / Gamma_new) -
                          (phi_trial - phi_old)^2 / (4 * dt) *
                          (1 / Gamma_new - 1 / Gamma_old)

            if log(rand()) < -dH + log_q_ratio
                phi_new[i+1, j+1, k+1] = phi_trial
            end
        end
    end

    return phi_new
end