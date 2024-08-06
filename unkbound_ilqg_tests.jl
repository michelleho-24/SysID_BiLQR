
include("ilqr_types.jl")
include("UnkBoundPOMDP.jl")
include("ilqg.jl")

mdp = UnkBoundPOMDP()

b0 = [mdp.s_init..., vec(1e-3 * Matrix{Float64}(I, (num_states(mdp)), (num_states(mdp))))...]
#TODO: add for loop with belief update later
a, info_dict = iLQG(mdp, b0)
println("action ", a)
println("cost ", info_dict[:cost])
