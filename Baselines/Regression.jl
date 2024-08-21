
# Import required libraries
using LinearAlgebra
using Convex
using SCS

# Function to generate synthetic data for a fluid system
function generate_data(A, B, x0, u, n_steps, process_noise=0.01)
    n_states = size(A, 1)
    X = zeros(n_states, n_steps)
    X[:, 1] = x0

    for t in 1:n_steps-1
        X[:, t+1] = A * X[:, t] + B * u[:, t] + process_noise * randn(n_states)
    end

    return X
end

# Define system matrices (A, B) for a simple fluid system
A = [0.8 0.1; -0.2 0.9]
B = [0.1; 0.05]
n_states = size(A, 1)
n_controls = size(B, 2)

# Generate random control inputs
n_steps = 100
u = randn(n_controls, n_steps)

# Initial state
x0 = randn(n_states)

# Generate synthetic data
X = generate_data(A, B, x0, u, n_steps)

# Set up the data matrices for DMDc (state and control inputs)
X_p = X[:, 1:end-1]   # past states
X_f = X[:, 2:end]     # future states
U = u[:, 1:end-1]     # control inputs

# Solve for A and B using least squares (DMDc regression)
Z = vcat(X_p, U)
Theta = X_f * pinv(Z)

# Extract A and B estimates from Theta
A_est = Theta[:, 1:n_states]
B_est = Theta[:, n_states+1:end]

println("Estimated A: ", A_est)
println("Estimated B: ", B_est)

# Construct the covariance matrix and perform optimal input design
function optimal_input_design(Z, sigma2)
    # Covariance matrix
    Gamma = sigma2 * inv(Z * Z')

    # Convex optimization problem to minimize the trace of the covariance matrix
    U_opt = Variable(size(U))
    objective = minimize(tr(Gamma))
    problem = Convex.solve!(objective, SCS.Optimizer)
    
    return U_opt.value
end

# Perform optimal input design
sigma2 = 0.01
optimal_u = optimal_input_design(Z, sigma2)
println("Optimal control input design: ", optimal_u)
