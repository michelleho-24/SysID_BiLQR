
include("ilqr_types.jl")
include("UnkBoundPOMDP.jl")
include("ilqg.jl")
include("ekf.jl")

mdp = UnkBoundPOMDP()

b0 = [mdp.s_init..., vec(1e-3 * Matrix{Float64}(I, (num_states(mdp)), (num_states(mdp))))...]

iters = 10
b = b0
s_bar = zeros(num_states(mdp), iters)

for i in 1:iters
    global b, s_bar
    a, info_dict = iLQG(mdp, b)
    println("action ", a)
    println("cost ", info_dict[:cost])
    s_bar = info_dict[:s_bar]

    # # dummy action and transition
    # a = randn(num_actions(mdp))
    # println("action ", a)
    # s = dyn_mean(mdp, b[1:4], a)
    # s_bar[:, i] = s

    # # get first state after action 
    # s_bar = info_dict[:s_bar]
    # s = s_bar[:, 2]

    # # get observation
    # z = obs_mean(mdp, s)

    # # update belief 
    # # this should be using the regular jacobians, not super 
    # b = ekf(mdp, b, a, z)

end 
