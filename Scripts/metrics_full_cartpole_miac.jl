using Statistics 
using JLD2
using LinearAlgebra
using SparseArrays
include("../Cartpole/cartpole_miac.jl")

pomdp = CartpoleMDP()

Σ_true = 2.0

function rmse(P_pred, P_true = Σ_true)
    # Ensure both matrices are square and diagonal
    # P_pred = Diagonal(P_pred)
    @assert size(P_pred) == size(P_true) "Matrices must have the same size"
    
    # Extract diagonal elements
    # diag_pred = diagm(P_pred)
    # diag_true = diagm(P_true)
    diag_pred = P_pred
    diag_true = P_true
    
    # Compute the squared differences
    squared_diff = (diag_pred .- diag_true).^2
    
    # Compute the mean squared error (MSE)
    mse = mean(squared_diff)
    
    # Compute and return the root mean squared error (RMSE)
    return sqrt(mse)
end

# function state_cost(state_vec)
#     γ = 0.9 
#     Q = pomdp.Q
#     cost = 0

#     for t in 1:length(state_vec)
#         s = state_vec[t]
#         cost += -γ^t * (s' * Q * s)
#     end

#     return cost 
# end 

function cost(states)
# function cost(all_s, u)
    n_states = num_states(pomdp)
    num_belief_states = n_states + n_states^2

    Q = spzeros(num_belief_states, num_belief_states)
    Q[1:n_states, 1:n_states] .= pomdp.Q
    R = pomdp.R
    Q_N = spzeros(num_belief_states, num_belief_states)
    Q_N[n_states + 1:end, n_states + 1:end] .= pomdp.Λ
    Q_N[1:n_states, 1:n_states] .= pomdp.Q_N
    s_goal = pomdp.s_goal

    cost = 0

    for s in states[1:end-1]
        println(s)
        println(s_goal)
        println(size(Q))
        cost += (s - s_goal)' * Q * (s - s_goal) # + u' * R * u + (s - s_goal)' * Q_N * (s - s_goal)
    end
    cost += (all_s[end]-s_goal)' * Q_N * (all_s[end]-s_goal)

    return cost
    # # Compute the cost of a state-action pair
    # return (s - s_goal)' * Q * (s - s_goal) # + u' * R * u + (s - s_goal)' * Q_N * (s - s_goal)
end

@load "bilqr_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ all_s

trace_bilqr_avg = mean([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
trace_bilqr_std = std([tr(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
println("BiLQR Σ Trace: ", trace_bilqr_avg, " ± ", trace_bilqr_std/length(all_ΣΘΘ))

# find average rmse and std
rmse_bilqr_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)]) 
rmse_bilqr_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
println("BiLQR RMSE: ", rmse_bilqr_avg, " ± ", rmse_bilqr_std/length(all_ΣΘΘ))

cost_bilqr_avg = mean([cost(states) for states in values(all_b)])
cost_bilqr_std = std([cost(states) for states in values(all_b)])
println("BiLQR Cost: ", cost_bilqr_avg, " ± ", cost_bilqr_std/length(all_b))


# # find expected value of the cost 

# @load "mpc_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ
# trace_mpc_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + mean([state_cost(s) for s in values(all_s)])
# trace_mpc_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + std([state_cost(s) for s in values(all_s)])

# rmse_mpc_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
# rmse_mpc_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])

# @load "random_cartpolefull_miac_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ

# trace_random_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + mean([state_cost(s) for s in values(all_s)])
# trace_random_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + std([state_cost(s) for s in values(all_s)])

# rmse_random_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)]) + mean([state_cost(s) for s in values(all_s)])
# rmse_random_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)]) + std([state_cost(s) for s in values(all_s)])

# @load "reg_cartpolefull_miac_results.jld2" all_b_ends all_mp_estimates all_mp_variances all_ΣΘΘ

# trace_regression_avg = mean([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + mean([state_cost(s) for s in values(all_s)])
# trace_regression_std = std([ΣΘΘ for ΣΘΘ in values(all_ΣΘΘ)]) + std([state_cost(s) for s in values(all_s)])

# rmse_regression_avg = mean([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])
# rmse_regression_std = std([rmse(ΣΘΘ) for ΣΘΘ in values(all_ΣΘΘ)])

# println("Trace of ΣΘΘ")
# println("BILQR: ", trace_bilqr_avg, " ± ", trace_bilqr_std)
# println("MPC: ", trace_mpc_avg, " ± ", trace_mpc_std)
# println("Random: ", trace_random_avg, " ± ", trace_random_std)
# println("Regression: ", trace_regression_avg, " ± ", trace_regression_std)

# println("RMSE of ΣΘΘ")
# println("BILQR: ", rmse_bilqr_avg, " ± ", rmse_bilqr_std)
# println("MPC: ", rmse_mpc_avg, " ± ", rmse_mpc_std)
# println("Random: ", rmse_random_avg, " ± ", rmse_random_std)
# println("Regression: ", rmse_regression_avg, " ± ", rmse_regression_std)

# # # Plot the estimated mass of the pole over time
# # plot(time_steps, mp_estimates, ribbon=sqrt.(mp_variances), label="Estimated mp ± 1 std dev", xlabel="Time Step", ylabel="Estimated mp", title="EKF Estimation of mp")
# # hline!([mp_true], label="True mp", linestyle=:dash)

# # # Show the plot
# # savefig("time_mp.png")  
