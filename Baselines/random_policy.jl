

function random_policy(pomdp, b0; N = 10, eps=1e-3, max_iters=1000)
   
    # select an action from the available range 
    # a = rand(rng, -10:10)
    a = 10*rand()
    return [a] 

end 

function cost(Q, R, Q_N, s, u, s_goal)
    # Compute the cost of a state-action pair
    return (s - s_goal)' * Q * (s - s_goal) + u' * R * u + (s - s_goal)' * Q_N * (s - s_goal)
end