using JLD2
include("../XPlane/xplane_sysid_tests_partial.jl")
# include("../Cartpole/cartpole_sysid_tests_partial.jl")

# Initialize dictionaries to store outputs for each seed
all_b_ends = Dict{Int, Vector{Vector{Float64}}}()
all_AB_estimates = Dict{Int, Vector{Vector{Float64}}}()
all_AB_variances = Dict{Int, Vector{Matrix{Float64}}}()
all_ΣΘΘ = Dict{Int, Matrix{Float64}}()
all_s = Dict{Int, Vector{Vector{Float64}}}()
all_u = Dict{Int, Vector{Vector{Float64}}}()
all_ABtrue = Dict{Int, Vector{Float64}}()
# all_Atrue = Dict{Int, Matrix{Float64}}()
# all_Btrue = Dict{Int, Matrix{Float64}}()

method = "random"

jld2_file = "$(method)_xplanepartial_sysid_results.jld2"
if isfile(jld2_file)
    @load jld2_file all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 
end

# Run the system identification experiment
for seed in 1:60 # skip 24, 25, 28, 29, 32, 33, 41, 42, 45, 48, 54, 56, 57, 
    println("Seed: ", seed)
    
    results = system_identification(seed, method)
    if results === nothing
        continue
    end
    b_end_seed, AB_vec_estimates_seed, AB_variances, ΣΘΘ_seed, s_seed, u_seed, AB_true_seed = results

    # Store the vector with the seed as the key
    # Store the vector with the seed as the key
    all_b_ends[seed] = b_end_seed
    all_AB_estimates[seed] = AB_vec_estimates_seed
    all_AB_variances[seed] = AB_variances   
    all_ΣΘΘ[seed] = ΣΘΘ_seed
    all_s[seed] = s_seed
    all_u[seed] = u_seed
    all_ABtrue[seed] = AB_true_seed

    # Save all dictionaries to a JLD2 file
    @save jld2_file all_b_ends all_AB_estimates all_AB_variances all_ΣΘΘ all_s all_u all_ABtrue 
end
