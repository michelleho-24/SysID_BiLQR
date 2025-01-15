

## POMDP type

abstract type iLQRPOMDP{S,A,O} <: POMDP{S,A,O} end

# interface
"""
    dyn_mean(p::iLQGPOMDP, s::AbstractVector, a::AbstractVector)::AbstractVector

    Return the mean dynamics update from state `s` with control `a`.
"""
function dyn_mean end

"""
    dyn_noise(p::iLQGPOMDP, s::AbstractVector, a::AbstractVector)::AbstractMatrix

    Return the covariance of the dynamics update from state `s` with control `a`.
"""
function dyn_noise end

"""
    obs_mean(p::iLQGPOMDP, sp::AbstractVector)::AbstractVector

    Return the mean observation from state `sp`.
"""
function obs_mean end

"""
    obs_noise(p::iLQGPOMDP, sp::AbstractVector)::AbstractMatrix

    Return the covariance of the observation from state `sp`.
"""
function obs_noise end 

"""
    num_states(p::iLQGPOMDP)::Int

    Return the dimensionality of the state space in the POMDP.
"""
function num_states end

"""
    num_actions(p::iLQGPOMDP)::Int

    Return the dimensionality of the action space in the POMDP.
"""
function num_actions end

"""
    num_observations(p::iLQGPOMDP)::Int

    Return the dimensionality of the observation space in the POMDP.
"""
function num_observations end

"""
    num_sysvars(p::iLQGPOMDP)::Int

    Return the dimensionality of the system variables in the POMDP.
"""
function num_sysvars end

POMDPs.transition(p::iLQRPOMDP, s, a) = MvNormal(dyn_mean(p, s, a), dyn_noise(p, s, a))
POMDPs.observation(p::iLQRPOMDP, s, a) = MvNormal(obs_mean(p, s, a), obs_noise(p, s, a))

### consistent policy interface for later user
