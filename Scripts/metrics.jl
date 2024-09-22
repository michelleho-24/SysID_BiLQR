

# # RMSE
# RMSE = sqrt(1/(num_states(mdp)*(num_states(mdp) + num_actions(mdp)))*tr(ΣΘΘ))

# println("Trace of Covariance: ", tr(ΣΘΘ))
# println("RMSE: ", RMSE)

# # Plot the estimated mass of the pole over time
# plot(time_steps, mp_estimates, ribbon=sqrt.(mp_variances), label="Estimated mp ± 1 std dev", xlabel="Time Step", ylabel="Estimated mp", title="EKF Estimation of mp")
# hline!([mp_true], label="True mp", linestyle=:dash)

# # Show the plot
# savefig("time_mp.png")  
