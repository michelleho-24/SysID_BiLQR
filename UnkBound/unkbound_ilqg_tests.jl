
include("../iLQG/ilqr_types.jl")
include("UnkBoundPOMDP.jl")
include("../iLQG/ilqg.jl")
include("../iLQG/ekf.jl")

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
    # s_bar = info_dict[:s_bar]

    # # dummy action 
    # a = randn(num_actions(mdp))
    # println("action ", a)

    # transitions
    s = dyn_mean(mdp, b[1:num_states(mdp)], a)
    s_bar[:, i] = s
    println("state ", s[1:num_states(mdp)])

    # implement terminal condition break
    tol = 1e-3  # Define your tolerance value

    if all(abs(s[i] - mdp.s_goal[i]) <= tol for i in 1:num_states(mdp))
        println("goal reached")
        break
    end

    # # get first state after action 
    # s_bar = info_dict[:s_bar]
    # s = s_bar[:, 2]

    # get observation
    z = obs_mean(mdp, s)

    # update belief 
    # this should be using the regular jacobians, not super 
    b = ekf(mdp, b, a, z)

end 
