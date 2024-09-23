using JLD2
include("../Cartpole/cartpole_sysid_tests.jl")
# include("../Cartpole/cartpole_sysid_tests_partial.jl")

# Initialize dictionaries to store outputs for each seed
all_b = Dict{Int, Vector{Float64}}()
all_mp_estimates = Dict{Int, Vector{Float64}}()
all_mp_variances = Dict{Int, Vector{Float64}}()
all_ΣΘΘ = Dict{Int, Float64}()


# Run the system identification experiment
for seed in 1:50
    println("Seed: ", seed)
    
    b_seed, mp_estimates_seed, mp_variances_seed, ΣΘΘ_seed = system_identification(seed)

    # Store the vector with the seed as the key
    all_b[seed] = b_seed
    all_mp_estimates[seed] = mp_estimates_seed
    all_mp_variances[seed] = mp_variances_seed
    all_ΣΘΘ[seed] = ΣΘΘ_seed

    # Save all dictionaries to a JLD2 file
    @save "mpc_cartpolefull_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ
end

# # Load the dictionaries from the JLD2 file
# @load "bilqr_sysid_results.jld2" all_b all_mp_estimates all_mp_variances all_ΣΘΘ

# # Access the vector for a specific seed
# seed = 1234
# println(all_ΣΘΘ[seed])

# Now you can use `vector` for further analysis