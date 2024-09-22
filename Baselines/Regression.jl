using LinearAlgebra


function regression(state, mp_belief)
    # Extract data from the current state (you'll need to customize this)
    # Assuming state = [x, θ, dx, dθ, mp] for the cart-pole
    x, θ, dx, dθ = state[1:4]
    g = 9.81  # Gravitational constant
    l = 1.0   # Length of the pole (assumed known)
    mc = 1.0  # Mass of the cart (assumed known)

    # Formulate system dynamics equation for the least squares fit
    # Using dynamics equations for dx/dt (cart acceleration) and dθ/dt (angular acceleration)
    sinθ = sin(θ)
    cosθ = cos(θ)

    # Observations (this is a simplified example, you'll need actual observations)
    observed_acceleration = dx  # Replace with actual observed accelerations
    observed_angular_acceleration = dθ  # Replace with actual angular accelerations

    # Construct A matrix and b vector based on the dynamic equation:
    # observed_acceleration = (mp * sinθ * (l * dθ^2 + g * cosθ) + a) / (mc + mp * sinθ^2)
    # b is the observation (dx or dθ), and A is the matrix of coefficients
    A = [
        sinθ * (l * dθ^2 + g * cosθ) / (mc + mp_belief * sinθ^2)
    ]

    # Observations (b vector) would be the actual values you're trying to fit to
    b = [observed_acceleration]

    # Solve least squares: Ax ≈ b to find mp
    # Normal equation: A' * A * mp ≈ A' * b
    mp_estimate = A \ b  # Solves the least squares problem

    return mp_estimate[1]  # Return the estimated mp
end
