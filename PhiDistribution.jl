# PhiDistribution.jl
#
# Tracks P(phi, t) after switching on the nonlinear-Gamma update.
#
# Protocol, for each independent replica:
#   1. Thermalize with a constant-Gamma Metropolis chain. Its proposal is
#      symmetric, so its stationary distribution is exactly exp(-H).
#   2. Keep running that chain to measure the baseline P0(phi).
#   3. Switch to ModelA_update from ModelA.jl and accumulate P(phi) in
#      logarithmically spaced time windows.
#
# Run with: julia --threads=auto PhiDistribution.jl

using Plots
using Printf
using Random
using Statistics

include("ModelA.jl")

# Constant-Gamma update, used only to prepare the exp(-H) ensemble.
function reference_update(phi::Array{Float64,3}, h::Float64, dt::Float64, L::Int,
                          m2::Float64, Gamma::Float64)

    phi_new = copy(phi)
    width   = sqrt(2 * Gamma * dt)

    for parity in [0, 1]
        for (i, j, k) in sites_with_parity(parity, L, L, L)

            phi_trial = phi_new[i+1, j+1, k+1] + width * randn()
            dH = delta_H(i, j, k, phi_new, phi_trial, h, L, m2)

            if dH <= 0 || rand() < exp(-dH)
                phi_new[i+1, j+1, k+1] = phi_trial
            end
        end
    end

    return phi_new
end

function add_to_histogram!(counts::AbstractVector{Float64}, phi::Array{Float64,3},
                           phi_min::Float64, bin_width::Float64)
    n_bins = length(counts)
    for value in phi
        bin = floor(Int, (value - phi_min) / bin_width) + 1
        if 1 <= bin <= n_bins
            counts[bin] += 1
        end
    end
    return counts
end

function run_replica(seed::Int, L::Int, m2::Float64, h::Float64, dt::Float64,
                     ref_gamma::Float64, n_therm::Int, n_baseline::Int,
                     window_edges::Vector{Int}, phi_min::Float64,
                     bin_width::Float64, n_bins::Int)

    Random.seed!(seed)

    n_windows = length(window_edges) - 1
    n_evolve  = window_edges[end]

    phi = zeros(Float64, L, L, L)

    # 1. Thermalize with the exact exp(-H) chain.
    phi2_therm = zeros(Float64, n_therm)
    for step in 1:n_therm
        phi = reference_update(phi, h, dt, L, m2, ref_gamma)
        phi2_therm[step] = mean(abs2, phi)
    end

    # 2. Measure the baseline distribution with the same chain.
    baseline  = zeros(Float64, n_bins)
    phi2_base = 0.0
    for step in 1:n_baseline
        phi = reference_update(phi, h, dt, L, m2, ref_gamma)
        add_to_histogram!(baseline, phi, phi_min, bin_width)
        phi2_base += mean(abs2, phi)
    end
    phi2_base /= n_baseline

    # 3. Switch on the nonlinear-Gamma update.
    histograms = zeros(Float64, n_bins, n_windows)
    phi2_t     = zeros(Float64, n_evolve)
    window     = 1
    for step in 1:n_evolve
        phi = ModelA_update(phi, h, dt, L, m2)
        phi2_t[step] = mean(abs2, phi)

        while step > window_edges[window + 1]
            window += 1
        end
        add_to_histogram!(view(histograms, :, window), phi, phi_min, bin_width)
    end

    return (phi2_therm = phi2_therm, baseline = baseline, phi2_base = phi2_base,
            histograms = histograms, phi2_t = phi2_t)
end

function main()
    # Model parameters
    L  = 32
    m2 = -2.29
    h  = 0.0
    dt = 0.02

    # Constant Gamma for the preparation chain; does not affect exp(-H).
    ref_gamma = 1.0

    # Run lengths, in calls to the update (time = steps * dt)
    n_replicas = 16
    n_therm    = 5000
    n_baseline = 1000
    window_edges = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]

    # Histogram range
    phi_min   = -4.0
    phi_max   =  4.0
    bin_width =  0.1

    n_bins    = round(Int, (phi_max - phi_min) / bin_width)
    n_windows = length(window_edges) - 1
    n_evolve  = window_edges[end]
    centers   = [phi_min + (b - 0.5) * bin_width for b in 1:n_bins]

    results = Vector{Any}(undef, n_replicas)
    Threads.@threads for r in 1:n_replicas
        results[r] = run_replica(1000 + r, L, m2, h, dt, ref_gamma, n_therm,
                                 n_baseline, window_edges, phi_min, bin_width,
                                 n_bins)
    end

    # Thermalization check: compare the last two quarters of the preparation.
    q3 = (div(n_therm, 2) + 1):div(3 * n_therm, 4)
    q4 = (div(3 * n_therm, 4) + 1):n_therm
    phi2_q3 = [mean(res.phi2_therm[q3]) for res in results]
    phi2_q4 = [mean(res.phi2_therm[q4]) for res in results]
    stderr_of(values) = std(values) / sqrt(length(values))
    @printf("Thermalization <phi^2>: third quarter %.5f +- %.5f, last quarter %.5f +- %.5f\n",
            mean(phi2_q3), stderr_of(phi2_q3), mean(phi2_q4), stderr_of(phi2_q4))

    # Combine replicas and normalize.
    baseline_counts = sum(res.baseline for res in results)
    window_counts   = sum(res.histograms for res in results)

    expected_baseline = n_replicas * n_baseline * L^3
    @printf("Fraction of baseline samples outside the histogram range: %.3e\n",
            1 - sum(baseline_counts) / expected_baseline)

    P0 = baseline_counts ./ (sum(baseline_counts) * bin_width)
    P  = similar(window_counts)
    for w in 1:n_windows
        P[:, w] = window_counts[:, w] ./ (sum(window_counts[:, w]) * bin_width)
    end

    phi2_base_values = [res.phi2_base for res in results]
    phi2_base_mean   = mean(phi2_base_values)
    phi2_base_err    = stderr_of(phi2_base_values)

    phi2_matrix = reduce(hcat, [res.phi2_t for res in results])
    phi2_mean   = vec(mean(phi2_matrix, dims = 2))
    phi2_err    = vec(std(phi2_matrix, dims = 2)) ./ sqrt(n_replicas)
    times       = (1:n_evolve) .* dt

    late = (window_edges[end - 1] + 1):n_evolve
    phi2_late_values = [mean(res.phi2_t[late]) for res in results]
    @printf("<phi^2>: baseline %.5f +- %.5f, final window %.5f +- %.5f\n",
            phi2_base_mean, phi2_base_err, mean(phi2_late_values),
            stderr_of(phi2_late_values))

    # Plot
    colors = cgrad(:viridis, n_windows, categorical = true)
    labels = [@sprintf("t in (%.2f, %.2f]", window_edges[w] * dt, window_edges[w + 1] * dt)
              for w in 1:n_windows]

    p1 = plot(centers, P0; color = :black, linewidth = 2, label = "baseline exp(-H)",
              xlabel = "phi", ylabel = "P(phi)")
    for w in 1:n_windows
        plot!(p1, centers, P[:, w]; color = colors[w], label = labels[w])
    end

    p2 = plot(; xlabel = "phi", ylabel = "P(phi, t) - P0(phi)", legend = false)
    hline!(p2, [0.0]; color = :black, linewidth = 1)
    for w in 1:n_windows
        plot!(p2, centers, P[:, w] .- P0; color = colors[w])
    end

    p3 = plot(times, phi2_mean; ribbon = phi2_err, xscale = :log10,
              xlabel = "t since switching on Gamma(phi)", ylabel = "<phi^2>",
              label = "nonlinear Gamma")
    hline!(p3, [phi2_base_mean]; color = :black, linestyle = :dash,
           label = "baseline exp(-H)")

    figure = plot(p1, p2, p3; layout = (3, 1), size = (800, 1100), margin = 5Plots.mm)
    savefig(figure, "PhiDistribution.png")

    return nothing
end

main()