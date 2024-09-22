using Parameters
using LinearAlgebra
using Distributions
using Random
using Statistics
using Plots

Random.seed!(42)

@with_kw mutable struct XPlanePOMDP
    # Cost matrices
    Q::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11, 11)
    Q_N::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 11, 11)
    R::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 3, 3)
    Λ::Matrix{Float64} = 1e-3 * Matrix{Float64}(I, 121 + 33, 121 + 33)
    
    # Initial true state (s_true)
    s_true_init::Vector{Float64} = [0.0, 0.0, 1000.0, 100.0, π/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # True A and B matrices (unknown to estimator)
    A_true::Matrix{Float64} = rand(11, 11) * 0.01 + I  # Small random perturbation around identity
    B_true::Matrix{Float64} = rand(11, 3) * 0.01  # Small random values
    
    # Initial estimates of A and B (for belief)
    A_est_init::Matrix{Float64} = 0.01 * Matrix{Float64}(I, 11, 11)
    B_est_init::Matrix{Float64} = 0.01 * ones(11, 3)
    
    # Flatten A and B
    A_vec_init::Vector{Float64} = vec(A_est_init)
    B_vec_init::Vector{Float64} = vec(B_est_init)
    
    # Initial state (including estimates of A and B)
    s_init::Vector{Float64} = vcat(s_true_init, A_vec_init, B_vec_init)
    
    # Mechanics
    m::Float64 = 6500.0
    g::Float64 = 9.81
    δt::Float64 = 0.1

    # Noise
    W_process::Matrix{Float64} = Diagonal(fill(1e-4, 11 + 121 + 33))
    W_obs::Matrix{Float64} = 1e-2 * Matrix{Float64}(I, 8)  # Adjusted for 8 observations
end

function dyn_mean(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector)
    s_true = s[1:11]
    # Use true A and B for simulation (unknown to estimator)
    A = p.A_true
    B = p.B_true
    s_new_true = A * s_true + B * a
    # The estimator's belief about A and B remains in s[12:end]
    A_vec = s[12:11+121]
    B_vec = s[11+121+1:end]
    s_new = vcat(s_new_true, A_vec, B_vec)
    return s_new
end

function dyn_noise(p::XPlanePOMDP, s::AbstractVector, a::AbstractVector)
    return p.W_process
end

function obs_mean(p::XPlanePOMDP, s::AbstractVector)
    return s[1:8]  # We observe only the first 8 elements
end

function obs_noise(p::XPlanePOMDP, s::AbstractVector)
    return p.W_obs
end

function num_states(p::XPlanePOMDP)
    return 11 + 121 + 33
end

function num_actions(p::XPlanePOMDP)
    return 3
end

function num_observations(p::XPlanePOMDP)
    return 8  # We observe only 8 elements
end

# Initialize the POMDP and state
p = XPlanePOMDP()
s = copy(p.s_init)

# Number of iterations for simulation
num_iterations = 500

# Data storage
data_s_true = []  # List to store s_true at each time step
data_a = []       # List to store actions
data_s_new = []   # List to store s_new (next s_true)

for t in 1:num_iterations
    # Choose an action (e.g., random action)
    a = rand(3) * 2.0 - 1.0  # Random values between -1 and 1
    
    # Get current s_true
    s_true = s[1:11]
    
    # Compute next state
    s_next = dyn_mean(p, s, a)
    
    # Add process noise
    process_noise = rand(MvNormal(zeros(num_states(p)), p.W_process))
    s_next += process_noise
    
    # Generate observation
    o = obs_mean(p, s_next)
    observation_noise = rand(MvNormal(zeros(num_observations(p)), p.W_obs))
    o += observation_noise
    
    # Store data
    s_new_true = o  # Since o = s_next[1:8] + noise
    push!(data_s_true, s_true)
    push!(data_a, a)
    push!(data_s_new, s_new_true)  # Now s_new_true has 8 elements
    
    # Update state (estimator's belief about A and B remains)
    s = s_next
end

# Prepare data for regression
n = length(data_s_true)
X_data = zeros(n, 14)  # Each row is [s_true', a']
Y_data = zeros(n, 8)   # Each row is s_new (only 8 elements)

for i in 1:n
    s_true = data_s_true[i]
    a = data_a[i]
    s_new = data_s_new[i]  # Only first 8 elements
    X_data[i, :] = vcat(s_true, a)
    Y_data[i, :] = s_new
end

# Perform regression to estimate Theta = [A_partial B_partial]'
Theta = (X_data' * X_data) \ (X_data' * Y_data)

# Extract A and B estimates
Theta = Theta'  # Now Theta is 8 x 14
A_estimated_partial = Theta[:, 1:11]  # Estimated first 8 rows of A
B_estimated_partial = Theta[:, 12:14] # Estimated first 8 rows of B

# Initialize full A and B matrices
A_estimated = zeros(11, 11)
B_estimated = zeros(11, 3)

# Populate the estimated parts
A_estimated[1:8, :] = A_estimated_partial
B_estimated[1:8, :] = B_estimated_partial

# For the unobserved rows (9 to 11), we can keep initial estimates or zeros
# For simplicity, we'll use zeros
# A_estimated[9:11, :] = zeros(3, 11)
# B_estimated[9:11, :] = zeros(3, 3)

# Compute traces
trace_A = tr(A_estimated)
trace_B = tr(B_estimated)

println("Estimated trace of A: ", trace_A)
println("Estimated trace of B: ", trace_B)
println("Sum of traces (tr(A) + tr(B)): ", trace_A + trace_B)
