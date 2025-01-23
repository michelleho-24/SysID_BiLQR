println("Testing...")

pomdp = create_cartpole()

horizon = 100
N = 10
eps = 1e-6
max_iters = 100

policy = BiLQRPolicy(pomdp = pomdp, N = N, eps = eps, max_iters = max_iters)

belief_updater = EKFUpdater(pomdp)

# action = POMDPs.action(policy, policy.pomdp.s_init)
all_b, info_dict = simulate(horizon, policy, belief_updater)

