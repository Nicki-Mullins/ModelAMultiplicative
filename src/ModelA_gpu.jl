# ModelA_gpu.jl
#
# GPU version of the Metropolis-Hastings update in ModelA.jl.
# H[φ] = Σ_x [ 1/2 (∇φ)^2 + m²/2 φ² + λ/4 φ⁴ - h φ ]
# Nonlinear relaxation rate is used, GammaA = GammaA0 + GammaA2 phi^2
#
# The kernel layout follows src/simulation.jl of github.com/joott/modelA:
# one CUDA kernel per checkerboard colour, a grid-stride loop over the
# L^3/2 sites of that colour, random numbers drawn inside the kernel, and
# the field updated in place.
#
# Differences from ModelA.jl:
#   - phi is updated IN PLACE (and also returned, so
#     `phi = ModelA_update_hastings(phi, h, dt, L, m2)` still works).
#   - Site indices inside this file are 1-based.
#   - L must be even.
#   - Only the Hastings update is provided.
#
# The floating-point type follows the array: pass a CuArray{Float64,3} or
# a CuArray{Float32,3}. Passing a plain Array runs the same site update on
# CPU threads.

using CUDA

const lambd = 4.0
const GammaA0 = 1.0
const GammaA2 = 0.2

@inline function relaxation(phi, a, b)
    return a + b * phi^2
end

# Periodic neighbours for 1-based indices.
@inline NNp(n, L) = n % L + 1
@inline NNm(n, L) = (n + L - 2) % L + 1

@inline function delta_H(phi, x, y, z, phi_trial, h, L, m2, lam)

    @inbounds phi_old = phi[x, y, z]
    dphi  = phi_trial - phi_old
    dphi2 = phi_trial^2 - phi_old^2

    @inbounds neighbor_sum = phi[NNp(x, L), y, z] + phi[NNm(x, L), y, z] +
                             phi[x, NNp(y, L), z] + phi[x, NNm(y, L), z] +
                             phi[x, y, NNp(z, L)] + phi[x, y, NNm(z, L)]

    kinetic   = 3 * dphi2 - dphi * neighbor_sum
    potential = m2 * dphi2 / 2 + lam * (phi_trial^4 - phi_old^4) / 4
    source    = -h * dphi

    return kinetic + potential + source
end

# Site number l (0-based, 0 <= l < L^3/2) of the given colour -> 1-based
# (x, y, z). The colour is the parity of the 0-based x + y + z.
@inline function checkerboard_site(l, parity, L)
    i = l ÷ L^2
    j = (l ÷ L) % L
    k = l % L
    return (2i + j + k + parity) % L + 1, j + 1, k + 1
end

@inline function hastings_site_update!(phi, x, y, z, L, m2, h, lam, dt, g0, g2)
    T = eltype(phi)

    @inbounds phi_old = phi[x, y, z]
    Gamma_old = relaxation(phi_old, g0, g2)

    # Box-Muller normal. 1 - rand(T) lies in (0, 1], so the log is finite.
    xi = cos(2 * T(pi) * rand(T)) * sqrt(-2 * log(1 - rand(T)))

    phi_trial = phi_old + sqrt(2 * Gamma_old * dt) * xi
    Gamma_new = relaxation(phi_trial, g0, g2)

    dH = delta_H(phi, x, y, z, phi_trial, h, L, m2, lam)

    # log[ q(trial -> old) / q(old -> trial) ]
    log_q_ratio = log(Gamma_old / Gamma_new) / 2 -
                  (phi_trial - phi_old)^2 / (4 * dt) *
                  (1 / Gamma_new - 1 / Gamma_old)

    accept = log(1 - rand(T)) < -dH + log_q_ratio

    @inbounds phi[x, y, z] = ifelse(accept, phi_trial, phi_old)
    return nothing
end

function check_lattice(phi, L)
    size(phi) == (L, L, L) || error("phi has size $(size(phi)), expected ($L, $L, $L)")
    iseven(L) || error("L = $L is odd; the checkerboard update needs even L")
    return nothing
end

# ----------------------------------------------------------------------
# GPU
# ----------------------------------------------------------------------

function half_sweep_kernel!(phi, parity, L, m2, h, lam, dt, g0, g2)
    index  = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x

    for l in index:stride:(L^3 ÷ 2 - 1)
        x, y, z = checkerboard_site(l, parity, L)
        hastings_site_update!(phi, x, y, z, L, m2, h, lam, dt, g0, g2)
    end
    return nothing
end

# Compiled kernel and launch configuration, per array type and L.
const kernel_cache = Dict{Tuple{DataType,Int},Any}()

function get_kernel(phi::CuArray{T,3}, L::Int) where {T}
    return get!(kernel_cache, (typeof(phi), L)) do
        z = zero(T)
        kernel  = @cuda launch=false half_sweep_kernel!(phi, 0, L, z, z, z, z, z, z)
        n_sites = L^3 ÷ 2
        config  = launch_configuration(kernel.fun)
        threads = min(n_sites, config.threads)
        blocks  = cld(n_sites, threads)
        (kernel, threads, blocks)
    end
end

function ModelA_update_hastings(phi::CuArray{T,3}, h, dt, L::Int, m2) where {T<:AbstractFloat}

    check_lattice(phi, L)
    kernel, threads, blocks = get_kernel(phi, L)

    for parity in 0:1
        kernel(phi, parity, L, T(m2), T(h), T(lambd), T(dt), T(GammaA0), T(GammaA2);
               threads = threads, blocks = blocks)
    end

    return phi
end

# ----------------------------------------------------------------------
# CPU (same site update, threaded over one colour at a time)
# ----------------------------------------------------------------------

function ModelA_update_hastings(phi::Array{T,3}, h, dt, L::Int, m2) where {T<:AbstractFloat}

    check_lattice(phi, L)
    m2_T, h_T, lam, dt_T = T(m2), T(h), T(lambd), T(dt)
    g0, g2 = T(GammaA0), T(GammaA2)

    for parity in 0:1
        Threads.@threads for l in 0:(L^3 ÷ 2 - 1)
            x, y, z = checkerboard_site(l, parity, L)
            hastings_site_update!(phi, x, y, z, L, m2_T, h_T, lam, dt_T, g0, g2)
        end
    end

    return phi
end