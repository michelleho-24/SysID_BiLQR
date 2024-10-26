using LinearAlgebra
using ForwardDiff
using Distributions
using POMDPs
using Infiltrator
using SparseArrays

form_belief_vector(x, Σ_diag) = [x...; Σ_diag...]

function sigma_pred(pomdp, x::AbstractVector, u::AbstractVector, Σ_diag::AbstractVector)
    At = ForwardDiff.jacobian(x -> dyn_mean(pomdp, x, u), x)
    # return At * Σ * At' + dyn_noise(pomdp, x, u)
    return diag(At * diagm(Σ_diag) * At' + dyn_noise(pomdp, x, u))
end

function sigma_update(pomdp, x::AbstractVector, Σ_diag::AbstractVector)
    Ct = ForwardDiff.jacobian(x -> obs_mean(pomdp, x), x)
    S = Ct * diagm(Σ_diag) * Ct' + obs_noise(pomdp, x)
    if any(isnan, S) || abs(det(S)) < 1e-12
        println("BiLQR S is nan, next seed...")
        return nothing
    end
    K = diagm(Σ_diag) * Ct' * inv(S)
    Σ_new = (I - K * Ct) * diagm(Σ_diag)
    return diag(Σ_new)  # Return only the diagonal elements
end

function update_belief(pomdp::iLQGPOMDP, belief::AbstractVector, u::AbstractVector)
    n_states = num_states(pomdp)
    x = belief[1:n_states]
    Σ_diag = belief[n_states+1:end]

    x_new = dyn_mean(pomdp, x, u)
    Σ_pred_diag = sigma_pred(pomdp, x, u, Σ_diag)
    Σ_new_diag = sigma_update(pomdp, x_new, Σ_pred_diag)

    if Σ_new_diag === nothing
        return nothing
    end

    return form_belief_vector(x_new, Σ_new_diag)
end

function superAB(pomdp, q, r, N, s_bar, u_bar)
    A = zeros(N, q, q)
    B = zeros(N, q, r)

    for k in 1:N
        A[k, :, :] = ForwardDiff.jacobian(bel -> update_belief(pomdp, bel, u_bar[k, :]), s_bar[k, :])
        B[k, :, :] = ForwardDiff.jacobian(u -> update_belief(pomdp, s_bar[k, :], u), u_bar[k, :])
    end

    return A, B
end

function cost(Q, R, Q_N, s, u, s_goal)
    return (s - s_goal)' * Q * (s - s_goal) + u' * R * u + (s - s_goal)' * Q_N * (s - s_goal)
end

function bilqr(pomdp, b0; N = 10, eps=1e-3, max_iters=50)
    if max_iters <= 1
        throw(ArgumentError("Argument `max_iters` must be at least 2."))
    end
    

    f = update_belief

    n_states = num_states(pomdp)
    num_belief_states = n_states + n_states

    s_goal = pomdp.s_goal

    Q = spzeros(num_belief_states, num_belief_states)
    Q[1:n_states, 1:n_states] .= pomdp.Q

    R = pomdp.R

    Q_N = spzeros(num_belief_states, num_belief_states)
    Q_N[n_states + 1:end, n_states + 1:end] .= pomdp.Λ
    Q_N[1:n_states, 1:n_states] .= pomdp.Q_N

    q = size(Q, 1)
    r = size(R, 1)

    Y = zeros(Float64, N, r, q)
    y = zeros(Float64, N, r)

    u_bar = zeros(Float64, N, r)
    s_bar = zeros(Float64, N+1, q)
    s_bar[1, :] = b0

    for k in 1:N
        next_belief = f(pomdp, s_bar[k, :], u_bar[k, :])
        if next_belief === nothing
            return nothing
        end
        s_bar[k+1, :] = next_belief
    end

    ds = zeros(Float64, N+1, q)
    du = zeros(Float64, N, r)

    converged = false

    for iter in 1:max_iters
        A, B = superAB(pomdp, q, r, N, s_bar, u_bar)

        V = copy(Q_N)
        v = Q_N * (s_bar[N+1, :] - s_goal)

        for k in N:-1:1
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

        s_bar_prev = copy(s_bar)
        for k in 1:N
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

    cost_final = cost(Q, R, Q_N, s_bar[N+1, :], u_bar[N, :], s_goal)

    info_dict = Dict(:converged => converged, :s_bar => s_bar, :u_bar => u_bar, :cost => cost_final)

    return u_bar[1, :], info_dict
end