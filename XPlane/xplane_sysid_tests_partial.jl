using Plots
using Random

include("../BiLQR/ilqr_types.jl")
include("../XPlane/Cessna_SysID_partial.jl")
include("../BiLQR/bilqr.jl")
include("../BiLQR/ekf.jl")
include("../Baselines/MPC.jl")
include("../Baselines/random_policy.jl")
# include("../Baselines/Regression.jl")

function xplane_sysid(seed, pomdp, b0, Σ0, iters = 50)
    Random.seed!(seed)
    # pomdp = XPlanePOMDP()

    # # want A and B to have higher uncertainty than the rest of the state
    # Σ0 = Diagonal(vcat(fill(0.01, 8), fill(0.1, 8^2 + 3*8)))
        
    # # Preallocate belief-state vector
    # b0 = Vector{Float64}(undef, length(mdp.s_init) + length(vec(Σ0)))
    # b0[1:length(mdp.s_init)] .= mdp.s_init
    # b0[length(mdp.s_init) + 1:end] .= vec(Σ0)
    # # Σ0 = Diagonal(vcat(fill(0.01, num_states(mdp)), fill(0.1, num_states(mdp)^2 + num_actions(mdp)*num_states(mdp))))
    # # b0 = [mdp.s_init..., vec(Σ0)...]
    # iters = 50
    b = b0

    # mp_cov_per_timestep = zeros(iters)
    # mp_var_per_timestep = zeros(iters)
    # cost_per_timestep = zeros(iters)
    A_estimates = Vector{Matrix{Float64}}(undef, iters)
    A_variances = Vector{Matrix{Float64}}(undef, iters)
    B_estimates = Vector{Matrix{Float64}}(undef, iters)
    B_variances = Vector{Matrix{Float64}}(undef, iters)
    AB_variances = Vector{Matrix{Float64}}(undef, iters)

    s_true = pomdp.s_init

    for t in 1:iters
        println("Belief Update Iteration: ", t)
        # global b, s_true

        # compute optimal action
        a, info_dict = bilqr(mdp, b)

        # vector of 3 actions between 0 and 10
        # a = [rand() * 10.0 for i in 1:3]

        # Simulate the true next state
        s_next_true = dyn_mean(pomdp, s_true, a)
        # println("s_next_true: ", s_next_true)

        # Add process noise to the true state
        noise_state = rand(MvNormal(mdp.W_state_process))
        noise_total = vcat(noise_state, vec(0.0 * Matrix{Float64}(I, 8, 8)), 
        vec(0.0 * Matrix{Float64}(ones(8, 3))))
        s_next_true += noise_total
        
        # Generate observation from the true next state
        z = obs_mean(pomdp, s_next_true)
        
        # Add observation noise
        obsnoise = rand(MvNormal(zeros(num_observations(pomdp)), pomdp.W_obs))
        z = z + obsnoise
        
        # Use your ekf function to update the belief
        b = ekf(pomdp, b, a, z)

        
        A_vec = b[8+1:8 + 8^2]
        B_vec = b[8 + 8^2 + 1:num_states(pomdp)]
        # AB_vec = vcat(A_vec, B_vec)
        cov_vec = b[num_states(pomdp)+1:end]

        # Reshape the flattened covariance into a 165x165 matrix
        cov_full = reshape(cov_vec, num_states(pomdp), num_states(pomdp))
        cov_A = cov_full[8+1:8^2+8, 8+1:8^2+8]
        cov_B = cov_full[8^2+8+1:8+8^2+3*8, 8^2+8+1:8+8^2+3*8]

        A_t = reshape(A_vec, 8, 8)
        B_t = reshape(B_vec, 8, 3)
        # AB_t = hcat(A_t, B_t)

        # Extract the 154x154 block corresponding to A (121 elements) and B (33 elements)
        # A occupies indices 12 through 132 (1-based)
        # B occupies indices 133 through 165
        cov_A_B = cov_full[11+1:num_states(pomdp),11+1:num_states(pomdp)]

        # Store estimates
        A_estimates[t] = A_t
        A_variances[t] = cov_A
        B_estimates[t] = B_t
        B_variances[t] = cov_B
        AB_variances[t] = cov_A_B
        # AB_estimates[t] = 
        
    end 

    ΣΘΘ = AB_variances[end]
    
    return b, A_estimates, A_variances, B_estimates, B_variances, AB_variances, ΣΘΘ 

    # println(tr(AB_variances[1]))

    # # plot trace of AB_variances over time
    # plot(1:iters, [tr(AB_variances[i]) for i in 1:iters], label="Trace of Covariance", xlabel="Time Step", ylabel="Trace of Covariance", title="Trace of Covariance over Time")
    # savefig("trace_covariance.png")

    # # Compute trace of covariance 
    # A_estimated = reshape(b[num_states(mdp)+1:num_states(mdp)+num_states(mdp)^2], num_states(mdp), num_states(mdp))
    # B_estimated = reshape(b[num_states(mdp)+num_states(mdp)^2+1:end], num_states(mdp), num_actions(mdp))

    # Θ = hcat(A_estimated, B_estimated)
    # tr_cov = tr(cov(Θ))
    # rmse = sqrt(1/(11*(11+3)) * tr_cov)

    # println("Trace of Covariance: ", tr_cov)
    # println("RMSE: ", RMSE)
end 