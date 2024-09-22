using Parameters
using LinearAlgebra
using Distributions
using Random
using Statistics  # For variance calculation
using Plots
include("../Cartpole/cartpole_sysid.jl")

Random.seed!(1)

# Initialize variables
pomdp = CartpoleMDP()

Σ0 = 0.01 * Matrix{Float64}(I, num_states(mdp), num_states(mdp))
# Increase uncertainty for mp
Σ0[end, end] = 0.1
b0 = [pomdp.s_init...; vec(Σ0)...]
b = b0

iters = 100
state = copy(pomdp.s_init)

# True mass of the pole (unknown to the estimator)
mp_true = 2.0  # True mass

# Data storage for regression
data_features = []
data_targets = []

# Simulation loop
for t in 1:iters
    global state 
    # Select an action (e.g., random force between -10 and 10)
    a = [rand() * 20.0 - 10.0]  # Random action between -10 and 10

    # Simulate the true next state (unknown to the estimator)
    s_true = copy(state)
    s_true[5] = mp_true  # Use true mp for simulation
    s_next_true = dyn_mean(mdp, s_true, a)
    # process_noise = rand(MvNormal(zeros(5), mdp.W_process))
    noise_state = rand(MvNormal(pomdp.W_state_process))
    noise_total = vcat(noise_state, 0.0)
    s_next_true += noise_total

    # Generate observation
    o = obs_mean(mdp, s_next_true)
    obsnoise = rand(MvNormal(zeros(4), pomdp.W_obs))
    o += obsnoise

    # Collect data for regression
    x, θ, dx, dθ = state[1:4]
    x_next, θ_next, dx_next, dθ_next = o

    # Compute observed acceleration
    x_acc_obs = (dx_next - dx) / pomdp.δt

    # Prepare features for linear regression
    # Attempt to linearize the dynamics equation w.r.t mp
    sinθ = sin(θ)
    cosθ = cos(θ)
    dθ2 = dθ^2
    total_mass = pomdp.mc + mp_true  # Using true mp for calculation

    # Approximate x_acc_obs = α * mp + β
    # Where α and β are functions of known variables
    # We rearrange the dynamics equation to get a linear relationship with mp

    # From the dynamics:
    # xacc = (a[1] + mp * l * dθ^2 * sinθ) / (mc + mp)
    #        - (mp * l * ((g * sinθ - cosθ * temp) / (l * (4/3 - mp * cosθ^2 / (mc + mp)))) * cosθ) / (mc + mp)
    # This is complex, but we can approximate or linearize around the current estimate

    # For simplicity, we'll use a simplified linear model:
    # x_acc_obs ≈ c1 * mp + c0
    # We'll compute c1 and c0 based on the current state and action

    # Let's define:
    c1 = (a[1] * (-1) - pomdp.l * dθ2 * sinθ * (dx_next - dx)) / ((pomdp.mc + mp_true) * pomdp.δt)
    c0 = (dx_next - dx) / pomdp.δt - a[1] / (pomdp.mc + mp_true)

    # Alternatively, collect the data and perform regression without attempting to linearize the dynamics
    # Due to the complexity, let's collect the data and perform regression at the end

    push!(data_features, [sinθ * dθ2 / total_mass])
    push!(data_targets, x_acc_obs - a[1] / total_mass)

    # Update state estimate for next iteration
    state = s_next_true
    state[5] = b[5]  # Use current estimate of mp
    state[2] = mod(state[2] + π, 2π) - π  # Keep θ within [-π, π]

end

# Convert data to matrices
X = hcat(data_features...)
X = reshape(X, :, 1)
y = data_targets

# Perform linear regression
# Model: y = θ * mp + ε
# θ is the coefficient, mp is the variable we want to estimate

# Add a column of ones if there is an intercept
X_design = X

# Perform linear regression
β_est = (X_design' * X_design) \ (X_design' * y)
mp_estimated = β_est[1]

# Estimate variance of mp
# Compute residuals
residuals = y .- X_design * β_est
σ² = sum(residuals.^2) / (length(y) - 1)
# Variance of β_est = σ² * (X'X)^(-1)
variance_mp = σ² * inv(X_design' * X_design)[1, 1]

# Update belief b
μ_new = copy(b[1:5])
μ_new[5] = mp_estimated
Σ_new = reshape(b[6:end], 5, 5)
Σ_new[5, 5] = variance_mp

# Pack the updated mean and covariance back into belief
b_new = [μ_new; vec(Σ_new)]
b = b_new

rmse = sqrt(mean((mp_true - mp_estimated)^2))

println("Estimated mp = $mp_estimated")
println("Variance of estimated mp = $variance_mp")
println("Final b[end] (variance of mp) = $(b[end])")
println("RMSE= $rmse")
