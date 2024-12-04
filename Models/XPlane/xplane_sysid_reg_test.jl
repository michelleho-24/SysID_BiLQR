using Parameters
using LinearAlgebra
using Distributions
using Random
using Statistics
using Plots

include("Cessna_SysID.jl")

Random.seed!(1234)

# Initialize the POMDP and state
mdp = XPlanePOMDP()
s = copy(p.s_init)

# Number of iterations for simulation
num_iterations = 500

# Data storage
data_s_true = []  # List to store s_true at each time step
data_a = []       # List to store actions
data_s_new = []   # List to store s_new (next s_true)

for t in 1:num_iterations
    global s

    # Choose an action (e.g., random action)
    a = rand(3) * 2.0 .- 1.0  # Random values between -1 and 1
    
    # Get current s_true
    s_true = s[1:11]
    
    # Compute next state
    s_next = dyn_mean(p, s, a)
    
    # Add process noise
    # process_noise = rand(MvNormal(zeros(num_states(p)), p.W_process))
    noise_state = rand(MvNormal(mdp.W_state_process))
    noise_total = vcat(noise_state, vec(0.0 * Matrix{Float64}(I, 11, 11)), 
    vec(0.0 * Matrix{Float64}(ones(11, 3))))
    s_next += noise_total
    
    # Generate observation
    o = obs_mean(p, s_next)
    observation_noise = rand(MvNormal(zeros(num_observations(p)), p.W_obs))
    o += observation_noise
    
    # Store data
    s_new_true = o  # Since o = s_next[1:11] + noise
    push!(data_s_true, s_true)
    push!(data_a, a)
    push!(data_s_new, s_new_true)
    
    # Update state (estimator's belief about A and B remains)
    s = s_next
end

# Prepare data for regression
n = length(data_s_true)
X_data = zeros(n, 14)  # Each row is [s_true', a']
Y_data = zeros(n, 11)  # Each row is s_new'

for i in 1:n
    s_true = data_s_true[i]
    a = data_a[i]
    s_new = data_s_new[i]
    X_data[i, :] = vcat(s_true, a)
    Y_data[i, :] = s_new
end

# Perform regression to estimate Theta = [A B]'
Theta = (X_data' * X_data) \ (X_data' * Y_data)

# Extract A and B estimates
Theta = Theta'  # Now Theta is 11 x 14
A_estimated = Theta[:, 1:11]
B_estimated = Theta[:, 12:14]

# Compute trace of covariance 
Θ = hcat(A_estimated, B_estimated)
tr_cov = tr(cov(Θ))

println("Trace of the covariance of theta: ", tr_cov)

rmse = sqrt(1/(11*(11+3)) * tr_cov)
println("RMSE: ", rmse)
