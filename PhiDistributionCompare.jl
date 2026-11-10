# PhiDistributionCompare.jl
#
# Compares how P(phi, t) changes under the Hastings and non-Hastings updates
# when both start from the same ensemble.
#
# Protocol, for each independent replica:
#   1. Thermalize with ModelA_update_hastings.
#   2. Keep running that chain to measure the baseline P0(phi).
#   3. From that same configuration, run two branches, one with
#      ModelA_update_hastings and one with ModelA_update, accumulating P(phi)
#      in logarithmically spaced time windows.
#
#
# Run with: julia --threads=auto PhiDistributionCompare.jl

using Plots
using Printf
using Random
using Statistics

include("ModelA.jl")

stderr_of(values) = std(values) / sqrt(length(values))

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

function evolve_branch(update, phi_start::Array{Float64,3}, h::Float64, dt::Float64,
                       L::Int, m2::Float64, window_edges::Vector{Int},
                       phi_min::Float64, bin_width::Float64, n_bins::Int)

    n_windows = length(window_edges) - 1
    n_evolve  = window_edges[end]

    phi        = copy(phi_start)
    histograms = zeros(Float64, n_bins, n_windows)
    phi2_t     = zeros(Float64, n_evolve)
    window     = 1

    for step in 1:n_evolve
        phi = update(phi, h, dt, L, m2)
        phi2_t[step] = mean(abs2, phi)

        while step > window_edges[window + 1]
            window += 1
        end
        add_to_histogram!(view(histograms, :, window), phi, phi_min, bin_width)
    end

    return (histograms = histograms, phi2_t = phi2_t)
end

function run_replica(seed::Int, L::Int, m2::Float64, h::Float64, dt::Float64,
                     n_therm::Int, n_baseline::Int, window_edges::Vector{Int},
                     phi_min::Float64, bin_width::Float64, n_bins::Int)

    Random.seed!(seed)

    phi = zeros(Float64, L, L, L)

    # 1. Thermalize with the Hastings update.
    phi2_therm = zeros(Float64, n_therm)
    for step in 1:n_therm
        phi = ModelA_update_hastings(phi, h, dt, L, m2)
        phi2_therm[step] = mean(abs2, phi)
    end

    # 2. Measure the baseline distribution with the same chain.
    baseline  = zeros(Float64, n_bins)
    phi2_base = 0.0
    for step in 1:n_baseline
        phi = ModelA_update_hastings(phi, h, dt, L, m2)
        add_to_histogram!(baseline, phi, phi_min, bin_width)
        phi2_base += mean(abs2, phi)
    end
    phi2_base /= n_baseline

    # 3. Branch from the same configuration.
    hastings = evolve_branch(ModelA_update_hastings, phi, h, dt, L, m2,
                             window_edges, phi_min, bin_width, n_bins)
    plain    = evolve_branch(ModelA_update, phi, h, dt, L, m2,
                             window_edges, phi_min, bin_width, n_bins)

    return (phi2_therm = phi2_therm, baseline = baseline, phi2_base = phi2_base,
            hastings = hastings, plain = plain)
end

# Normalizes each column of window counts to a probability density.
function normalize_windows(window_counts::Matrix{Float64}, bin_width::Float64)
    P = similar(window_counts)
    for w in 1:size(window_counts, 2)
        P[:, w] = window_counts[:, w] ./ (sum(window_counts[:, w]) * bin_width)
    end
    return P
end

function main()
    # Model parameters
    L  = 32
    m2 = -2.29
    h  = 0.0
    dt = 0.02

    # Run lengths, in calls to the update (time = steps * dt)
    n_replicas = 16
    n_therm    = 5000
    n_baseline = 4000
    window_edges = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]

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
        results[r] = run_replica(2000 + r, L, m2, h, dt, n_therm, n_baseline,
                                 window_edges, phi_min, bin_width, n_bins)
    end

    # Thermalization check: compare the last two quarters of the preparation.
    q3 = (div(n_therm, 2) + 1):div(3 * n_therm, 4)
    q4 = (div(3 * n_therm, 4) + 1):n_therm
    phi2_q3 = [mean(res.phi2_therm[q3]) for res in results]
    phi2_q4 = [mean(res.phi2_therm[q4]) for res in results]
    @printf("Thermalization <phi^2>: third quarter %.5f +- %.5f, last quarter %.5f +- %.5f\n",
            mean(phi2_q3), stderr_of(phi2_q3), mean(phi2_q4), stderr_of(phi2_q4))

    # Combine replicas and normalize.
    baseline_counts = sum(res.baseline for res in results)
    expected_baseline = n_replicas * n_baseline * L^3
    @printf("Fraction of baseline samples outside the histogram range: %.3e\n",
            1 - sum(baseline_counts) / expected_baseline)

    P0 = baseline_counts ./ (sum(baseline_counts) * bin_width)
    P_hastings = normalize_windows(sum(res.hastings.histograms for res in results), bin_width)
    P_plain    = normalize_windows(sum(res.plain.histograms for res in results), bin_width)

    D_hastings = P_hastings .- P0
    D_plain    = P_plain .- P0

    # <phi^2> summaries. Differences are taken per replica before averaging.
    phi2_base_values = [res.phi2_base for res in results]
    phi2_base_mean   = mean(phi2_base_values)
    @printf("<phi^2> baseline:             %.5f +- %.5f\n",
            phi2_base_mean, stderr_of(phi2_base_values))

    late = (window_edges[end - 1] + 1):n_evolve
    late_hastings = [mean(res.hastings.phi2_t[late]) for res in results]
    late_plain    = [mean(res.plain.phi2_t[late]) for res in results]

    @printf("<phi^2> final window Hastings: %.5f +- %.5f\n",
            mean(late_hastings), stderr_of(late_hastings))
    @printf("<phi^2> final window plain:    %.5f +- %.5f\n",
            mean(late_plain), stderr_of(late_plain))

    diff_hastings = late_hastings .- phi2_base_values
    diff_plain    = late_plain .- phi2_base_values
    diff_branches = late_plain .- late_hastings
    @printf("Final - baseline, Hastings:   %+.5f +- %.5f\n",
            mean(diff_hastings), stderr_of(diff_hastings))
    @printf("Final - baseline, plain:      %+.5f +- %.5f\n",
            mean(diff_plain), stderr_of(diff_plain))
    @printf("Plain - Hastings, final:      %+.5f +- %.5f\n",
            mean(diff_branches), stderr_of(diff_branches))

    phi2_hastings_matrix = reduce(hcat, [res.hastings.phi2_t for res in results])
    phi2_plain_matrix    = reduce(hcat, [res.plain.phi2_t for res in results])
    phi2_hastings_mean   = vec(mean(phi2_hastings_matrix, dims = 2))
    phi2_plain_mean      = vec(mean(phi2_plain_matrix, dims = 2))
    phi2_hastings_err    = vec(std(phi2_hastings_matrix, dims = 2)) ./ sqrt(n_replicas)
    phi2_plain_err       = vec(std(phi2_plain_matrix, dims = 2)) ./ sqrt(n_replicas)
    times                = (1:n_evolve) .* dt

    # Plot
    colors = cgrad(:viridis, n_windows, categorical = true)
    labels = [@sprintf("t in (%.2f, %.2f]", window_edges[w] * dt, window_edges[w + 1] * dt)
              for w in 1:n_windows]

    # Same vertical scale on every difference panel.
    y_max   = 1.05 * maximum(abs, vcat(vec(D_hastings), vec(D_plain)))
    y_range = (-y_max, y_max)

    p1 = plot(centers, P0; color = :black, linewidth = 2, label = "baseline",
              xlabel = "phi", ylabel = "P(phi)", title = "Distribution, final window")
    plot!(p1, centers, P_hastings[:, end]; color = :blue, label = "Hastings")
    plot!(p1, centers, P_plain[:, end]; color = :red, label = "non-Hastings")

    p2 = plot(; xlabel = "phi", ylabel = "P(phi, t) - P0(phi)",
              title = "Change from baseline, final window")
    hline!(p2, [0.0]; color = :black, linewidth = 1, label = "")
    plot!(p2, centers, D_hastings[:, end]; color = :blue, label = "Hastings")
    plot!(p2, centers, D_plain[:, end]; color = :red, label = "non-Hastings")

    p3 = plot(; xlabel = "phi", ylabel = "P(phi, t) - P0(phi)", title = "Hastings",
              ylims = y_range, legendfontsize = 6)
    hline!(p3, [0.0]; color = :black, linewidth = 1, label = "")
    for w in 1:n_windows
        plot!(p3, centers, D_hastings[:, w]; color = colors[w], label = labels[w])
    end

    p4 = plot(; xlabel = "phi", ylabel = "P(phi, t) - P0(phi)", title = "Non-Hastings",
              ylims = y_range, legend = false)
    hline!(p4, [0.0]; color = :black, linewidth = 1)
    for w in 1:n_windows
        plot!(p4, centers, D_plain[:, w]; color = colors[w])
    end

    p5 = plot(times, phi2_hastings_mean; ribbon = phi2_hastings_err, color = :blue,
              xscale = :log10, xlabel = "t since branching", ylabel = "<phi^2>",
              label = "Hastings")
    plot!(p5, times, phi2_plain_mean; ribbon = phi2_plain_err, color = :red,
          label = "non-Hastings")
    hline!(p5, [phi2_base_mean]; color = :black, linestyle = :dash, label = "baseline")

    figure = plot(p1, p2, p3, p4, p5; layout = @layout([a b; c d; e]),
                  size = (1200, 1100), margin = 5Plots.mm)
    savefig(figure, "PhiDistributionCompare.png")

    return nothing
end

main()