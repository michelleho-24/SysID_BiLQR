using JLD2
include("../Cartpole/cartpole_sysid_reg_test_partial.jl")
# include("../Cartpole/cartpole_sysid_tests_partial.jl")

# Initialize dictionaries to store outputs for each seed
all_b_ends = Dict{Int, Vector{Float64}}()
all_mp_estimates = Dict{Int, Vector{Float64}}()
all_mp_variances = Dict{Int, Vector{Float64}}()
all_ΣΘΘ = Dict{Int, Float64}()


# Run the system identification experiment
for seed in 1:50
    println("Seed: ", seed)
    
    b_end_seed, mp_estimates_seed, mp_variances_seed, ΣΘΘ_seed = regression(seed)

    # Store the vector with the seed as the key
    all_b_ends[seed] = b_end_seed
    all_mp_estimates[seed] = mp_estimates_seed
    all_mp_variances[seed] = mp_variances_seed
    all_ΣΘΘ[seed] = ΣΘΘ_seed

    # Save all dictionaries to a JLD2 file
    @save "reg_cartpolepartial_sysid_results.jld2" all_b_ends all_mp_estimates all_mp_variances all_ΣΘΘ
end
