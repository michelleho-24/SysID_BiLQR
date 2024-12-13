using LinearAlgebra
using ForwardDiff
using Distributions
using POMDPs
using SparseArrays

"""
    BiLQRPolicy(pomdp, N, eps, max_iters)

A policy that uses the Belief Iterative Linear Quadratic Regulator (BiLQR) algorithm
to compute actions for a given belief-based POMDP.

# Fields
- `pomdp`: The POMDP model.
- `N::Int`: The planning horizon.
- `eps::Float64`: Convergence tolerance for the algorithm.
- `max_iters::Int`: Maximum iterations for the iLQR loop.
"""
struct BiLQRPolicy <: POMDPs.Policy
    pomdp::iLQRPOMDP
    horizon::Int
    N::Int
    eps::Float64
    max_iters::Int
end

"""
    action_info(policy::BiLQRPolicy, b)

Compute the action using the BiLQR algorithm for the given belief.

# Arguments
- `policy::BiLQRPolicy`: The BiLQR policy.
- `b`: The belief vector.

# Returns
- The optimal action computed by the BiLQR algorithm.
"""
function action_info(policy::BiLQRPolicy, b)
    pomdp = policy.pomdp

    # Run the BiLQR algorithm
    u, action_info = bilqr(
        pomdp,
        b;
        N=policy.N,
        eps=policy.eps,
        max_iters=policy.max_iters,
    )

    # Handle cases where the algorithm fails (e.g., returns `nothing`)
    if u === nothing
        error("BiLQR failed to compute an action")
    end

    return u, action_info 
end

action_info(policy::BiLQRPolicy, b::GaussianBelief) = action_info(policy, b.mean)

"""
    bilqr(pomdp, b0; N, eps, max_iters)

Solve for the optimal action using the Belief Iterative Linear Quadratic Regulator (BiLQR) algorithm.

# Arguments
- `pomdp`: The POMDP model.
- `b0`: The initial belief state.
- `N::Int`: The planning horizon.
- `eps::Float64`: Convergence tolerance.
- `max_iters::Int`: Maximum iterations for the iLQR loop.

# Returns
- `u_bar[1,:]`: The optimal action for the initial belief.
- `info_dict`: Additional information about the optimization process.
"""
function bilqr(pomdp, b0; N = 10, eps=1e-3, max_iters=100)

    if max_iters <= 1
        throw(ArgumentError("Argument `max_iters` must be at least 2."))
    end

    # define the dynamics function
    f = update_belief # includes update for the mean AND covariance 

    n_states = num_states(pomdp)
    num_belief_states = n_states + n_states^2

    s_goal = pomdp.s_goal

    Q = spzeros(num_belief_states, num_belief_states)
    Q[1:n_states, 1:n_states] .= pomdp.Q

    R = pomdp.R

    Q_N = spzeros(num_belief_states, num_belief_states)
    Q_N[n_states + 1:end, n_states + 1:end] .= pomdp.Λ
    Q_N[1:n_states, 1:n_states] .= pomdp.Q_N

    q = size(Q,1)  # state dimension including means and covariances, was n
    r = size(R, 1)  # control dimension, was m, should be 2 

    # Initialize gains `Y` and offsets `y` for the policy
    Y = zeros(Float64, N, r, q)
    y = zeros(Float64, N, r)

    # Initialize the nominal trajectory `(s_bar, u_bar)`, and the deviations `(ds, du)`
    u_bar = zeros(Float64, N, r)
    s_bar = zeros(Float64, N+1, q)
    s_bar[1, :] = b0

    for k in 1:N
        next_belief = f(pomdp, s_bar[k, :], u_bar[k, :])
        if next_belief === nothing
            return nothing  # Or handle this in another way, e.g., skip the seed
        end
        s_bar[k+1, :] = next_belief
    end

    ds = zeros(Float64, N+1, q)
    du = zeros(Float64, N, r)

    # iLQR loop
    converged = false

    for iter in 1:max_iters

        A, B = superAB(pomdp, q, r, N, s_bar, u_bar)

        # Use Λ for the final state cost
        V = copy(Q_N)

        v = Q_N * (s_bar[N+1, :] - s_goal)


        # Backward pass
        for k in N:-1:1
            # println("Backward k: ", k)
            Qxk = Q * (s_bar[k, :] - s_goal) + A[k, :, :]' * v
            Quk = R * u_bar[k, :] + B[k, :, :]' * v
            Qxx = Q + A[k, :, :]' * V * A[k, :, :]
            Quu = R + B[k, :, :]' * V * B[k, :, :]
            Qux = B[k, :, :]' * V * A[k, :, :]

            y[k, :] = -inv(Quu) * Quk
            Y[k, :, :] = -inv(Quu) * Qux

            v = Qxk - Y[k, :, :]' * Quu * y[k, :]
            V = Qxx - Y[k, :, :]' * Quu * Y[k, :, :]
        end

        # Forward pass
        s_bar_prev = copy(s_bar)
        for k in 1:N
            # println("Forward k: ", k)
            du[k, :] = Y[k, :, :] * ds[k, :] + y[k, :]
            next_belief = f(pomdp, s_bar[k, :], u_bar[k, :] + du[k, :])
            if next_belief === nothing
                return nothing
            end 
            s_bar[k+1, :] = next_belief
            u_bar[k, :] += du[k, :]
            ds[k+1, :] = s_bar[k+1, :] - s_bar_prev[k+1, :]
        end

        if maximum(abs.(du)) < eps
            converged = true
            break
        end
    end

    # create random action to get random cost
    # u_rand = randn(r)
    # cost_final = cost(Q, R, Q_N, s_bar[N+1, :], u_rand, s_goal)
    cost_final = cost(Q, R, Q_N, s_bar[N+1, :], u_bar[N, :], s_goal)

    info_dict = Dict(:converged => converged, :s_bar => s_bar, :u_bar => u_bar, :cost => cost_final)
   
    return u_bar[1,:], info_dict
end
