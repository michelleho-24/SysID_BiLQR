using JLD2
include("../XPlane/xplane_sysid_tests_partial.jl")
# include("../Cartpole/cartpole_sysid_tests_partial.jl")

# Initialize dictionaries to store outputs for each seed
all_b_ends = Dict{Int, Vector{Float16}}()
all_A_estimates = Dict{Int,  Vector{Matrix{Float16}}}()
all_A_variances = Dict{Int,  Vector{Matrix{Float16}}}()
all_B_estimates = Dict{Int,  Vector{Matrix{Float16}}}()
all_B_variances = Dict{Int,  Vector{Matrix{Float16}}}()
all_AB_variances = Dict{Int,  Vector{Matrix{Float16}}}()
all_ΣΘΘ = Dict{Int, Matrix{Float16}}()

pomdp = XPlanePOMDP()

# want A and B to have higher uncertainty than the rest of the state
Σ0 = Diagonal(vcat(fill(0.01, 8), fill(0.1, 8^2 + 3*8)))
    
# Preallocate belief-state vector
b0 = Vector{Float16}(undef, length(mdp.s_init) + length(vec(Σ0)))
b0[1:length(mdp.s_init)] .= mdp.s_init
b0[length(mdp.s_init) + 1:end] .= vec(Σ0)

# Run the system identification experiment
for seed in 1:20
    global pomdp, b0, Σ0
    println("Seed: ", seed)
    
    b_end_seed, A_estimates_seed, A_variances_seed, B_estimates_seed, B_variances_seed, AB_variances_seed, ΣΘΘ_seed = xplane_sysid(seed, pomdp, b0)

    # Store the vector with the seed as the key
    all_b_ends[seed] = b_end_seed
    all_A_estimates[seed] = A_estimates_seed
    all_A_variances[seed] = A_variances_seed
    all_B_estimates[seed] = B_estimates_seed
    all_B_variances[seed] = B_variances_seed
    all_AB_variances[seed] = AB_variances_seed
    all_ΣΘΘ[seed] = ΣΘΘ_seed

    # Save all dictionaries to a JLD2 file
    @save "bilqr_xplanepartial_sysid_results.jld2" all_b_ends all_A_estimates all_A_variances all_B_estimates all_B_variances all_AB_variances all_ΣΘΘ
end

# # Load the dictionaries from the JLD2 file
# @load "bilqr_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ

# # Access the vector for a specific seed
# seed = 1234
# println(all_ΣΘΘ[seed])

# Now you can use `vector` for further analysis