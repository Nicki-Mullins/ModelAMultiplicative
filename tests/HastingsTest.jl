# HastingsTest.jl
#
# Checks that ModelA_update_hastings samples exp(-H) and is stable over a
# long run.
#
# For each replica, started either cold (phi = 0) or hot (phi ~ N(0,1)):
#   - evolve with ModelA_update_hastings only;
#   - every `measure_every` steps record
#       S1   = <phi dH/dphi>         (exactly 1 for exp(-H))
#       S3   = <phi^3 dH/dphi>       (exactly 3 <phi^2> for exp(-H))
#       phi2 = <phi^2>
#       M    = <phi>
#     where <.> is the average over sites.
#
# The run is split into blocks. The first `n_skip` blocks are discarded as
# thermalization; the rest give the estimates and a linear trend.
#
# The relaxation parameters are whatever is hardcoded in ModelA.jl.
#
# Run with: julia --threads=auto HastingsTest.jl

using Plots
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "ModelA.jl"))

stderr_of(values) = std(values) / sqrt(length(values))

# Site averages of phi*dH/dphi, phi^3*dH/dphi, phi^2 and phi.
# dH/dphi_x = (6 + m2) phi_x - (sum of 6 neighbors) + lambd phi_x^3 - h
function measure(phi::Array{Float64,3}, L::Int, m2::Float64, h::Float64)
    s1 = 0.0
    s3 = 0.0
    p2 = 0.0
    m  = 0.0
    for k in 1:L, j in 1:L, i in 1:L
        p  = phi[i, j, k]
        nn = phi[mod1(i+1, L), j, k] + phi[mod1(i-1, L), j, k] +
             phi[i, mod1(j+1, L), k] + phi[i, mod1(j-1, L), k] +
             phi[i, j, mod1(k+1, L)] + phi[i, j, mod1(k-1, L)]
        dH = (6 + m2) * p - nn + lambd * p^3 - h
        s1 += p * dH
        s3 += p^3 * dH
        p2 += p^2
        m  += p
    end
    V = L^3
    return s1 / V, s3 / V, p2 / V, m / V
end

function run_replica(seed::Int, hot::Bool, L::Int, m2::Float64, h::Float64,
                     dt::Float64, n_steps::Int, measure_every::Int)

    Random.seed!(seed)

    phi = hot ? randn(L, L, L) : zeros(Float64, L, L, L)

    n_meas = div(n_steps, measure_every)
    S1   = zeros(Float64, n_meas)
    S3   = zeros(Float64, n_meas)
    phi2 = zeros(Float64, n_meas)
    M    = zeros(Float64, n_meas)

    for n in 1:n_meas
        for _ in 1:measure_every
            phi = ModelA_update_hastings(phi, h, dt, L, m2)
        end
        S1[n], S3[n], phi2[n], M[n] = measure(phi, L, m2, h)
    end

    return (S1 = S1, S3 = S3, phi2 = phi2, M = M)
end

# Means of consecutive blocks of a time series.
function block_means(series::Vector{Float64}, n_blocks::Int)
    block_len = div(length(series), n_blocks)
    return [mean(view(series, (b - 1) * block_len + 1 : b * block_len))
            for b in 1:n_blocks]
end

# Least-squares slope of y against 1, 2, ..., length(y).
function slope(y::AbstractVector{Float64})
    x  = collect(1.0:length(y))
    xm = mean(x)
    ym = mean(y)
    return sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
end

# blocks_*: n_blocks x n_replicas matrices of block means.
function summarize(name::String, target, blocks_cold::Matrix{Float64},
                   blocks_hot::Matrix{Float64}, keep::UnitRange{Int})

    cold = vec(mean(blocks_cold[keep, :], dims = 1))
    hot  = vec(mean(blocks_hot[keep, :], dims = 1))
    both = vcat(cold, hot)

    # Total change of the linear fit across the kept blocks, per replica.
    span   = length(keep) - 1
    trends = vcat([span * slope(blocks_cold[keep, r]) for r in axes(blocks_cold, 2)],
                  [span * slope(blocks_hot[keep, r]) for r in axes(blocks_hot, 2)])

    println(name)
    if target === nothing
        @printf("  all replicas:  %.5f +- %.5f\n", mean(both), stderr_of(both))
    else
        @printf("  all replicas:  %.5f +- %.5f   (exact: %g)\n",
                mean(both), stderr_of(both), target)
    end
    @printf("  cold start:    %.5f +- %.5f\n", mean(cold), stderr_of(cold))
    @printf("  hot start:     %.5f +- %.5f\n", mean(hot), stderr_of(hot))
    @printf("  cold - hot:    %+.5f +- %.5f\n", mean(cold) - mean(hot),
            sqrt(stderr_of(cold)^2 + stderr_of(hot)^2))
    @printf("  linear trend:  %+.5f +- %.5f   (change across kept blocks)\n",
            mean(trends), stderr_of(trends))
    return nothing
end

function add_curves!(p, times, blocks_cold::Matrix{Float64}, blocks_hot::Matrix{Float64})
    n = size(blocks_cold, 2)
    plot!(p, times, vec(mean(blocks_cold, dims = 2));
          ribbon = vec(std(blocks_cold, dims = 2)) ./ sqrt(n),
          color = :blue, marker = :circle, markersize = 3, label = "cold start")
    n = size(blocks_hot, 2)
    plot!(p, times, vec(mean(blocks_hot, dims = 2));
          ribbon = vec(std(blocks_hot, dims = 2)) ./ sqrt(n),
          color = :red, marker = :circle, markersize = 3, label = "hot start")
    return p
end

function main()
    # Model parameters
    L  = 32
    m2 = -2.29
    h  = 0.0
    dt = 0.02

    # Run lengths
    n_replicas    = 8        # per start
    n_steps       = 80000
    measure_every = 50
    n_blocks      = 20
    n_skip        = 8        # blocks discarded as thermalization

    keep = (n_skip + 1):n_blocks

    runs = [(hot, r) for hot in (false, true) for r in 1:n_replicas]
    results = Vector{Any}(undef, length(runs))
    Threads.@threads for i in eachindex(runs)
        hot, r = runs[i]
        results[i] = run_replica(3000 + i, hot, L, m2, h, dt, n_steps, measure_every)
    end

    # Block means, as n_blocks x n_replicas matrices per start.
    function blocks_of(f, hot::Bool)
        columns = [block_means(f(results[i]), n_blocks)
                   for i in eachindex(runs) if runs[i][1] == hot]
        return reduce(hcat, columns)
    end

    quantities = [
        ("S1 = <phi dH>",                   1.0,     res -> res.S1),
        ("D3 = <phi^3 dH> - 3 <phi^2>",     0.0,     res -> res.S3 .- 3 .* res.phi2),
        ("<phi^2>",                         nothing, res -> res.phi2),
        ("<phi^2> - M^2",                   nothing, res -> res.phi2 .- res.M .^ 2),
        ("M^2",                             nothing, res -> res.M .^ 2),
    ]

    block_time = div(div(n_steps, measure_every), n_blocks) * measure_every * dt
    times = [(b - 0.5) * block_time for b in 1:n_blocks]

    @printf("L = %d, m2 = %.3f, h = %.3f, dt = %.3f, lambd = %.1f\n", L, m2, h, dt, lambd)
    @printf("%d steps (t = %.0f), %d replicas per start, blocks %d-%d of %d kept\n\n",
            n_steps, n_steps * dt, n_replicas, first(keep), last(keep), n_blocks)

    panels = []
    for (name, target, f) in quantities
        cold = blocks_of(f, false)
        hot  = blocks_of(f, true)
        summarize(name, target, cold, hot, keep)
        println()

        p = plot(; xlabel = "t", ylabel = name, legend = isempty(panels) ? :best : false)
        # Draw the data first so the axis limits come from the curves; an
        # hline!/vline! on an empty plot adds placeholder points at 1 and 2.
        add_curves!(p, times, cold, hot)
        if target !== nothing
            hline!(p, [target]; color = :black, linestyle = :dash, label = "exact")
        end
        push!(panels, p)
    end

    figure = plot(panels...; layout = (length(panels), 1), size = (800, 1500),
                  margin = 5Plots.mm, left_margin = 15Plots.mm)
    savefig(figure, "HastingsTest.png")

    return nothing
end

main()