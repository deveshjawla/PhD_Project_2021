using Distributions

d = 2;
n = 100;

observations = randn((d, n)); # 100 observations from 2D 𝒩(0, 1)

# Define generative model
#    μ ~ 𝒩(0, 1)
#    xᵢ ∼ 𝒩(μ, 1) , ∀i = 1, …, n
prior(μ) = logpdf(MvNormal(ones(d)), μ)

likelihood(x, μ) = sum(logpdf(MvNormal(μ, ones(d)), x))

logπ(μ) = likelihood(observations, μ) + prior(μ)

logπ(randn(2))  # <= just checking that it works

using DistributionsAD, AdvancedVI

# Using a function z ↦ q(⋅∣z)
getq(θ) = TuringDiagMvNormal(θ[1:d], exp.(θ[d+1:4]))

# Perform VI
advi = ADVI(10, 10_000)

q = vi(logπ, advi, getq, randn(4))
AdvancedVI.elbo(advi, q, logπ, 1000)

# True posterior
using ConjugatePriors

pri = MvNormal(zeros(2), ones(2));

true_posterior = posterior((pri, pri.Σ), MvNormal, observations)

using Plots

p_samples = rand(true_posterior, 10_000);
q_samples = rand(q, 10_000);

p1 = histogram(p_samples[1, :], label = "p");
histogram!(q_samples[1, :], alpha = 0.7, label = "q");

title!(raw"$\mu_1$")

p2 = histogram(p_samples[2, :], label = "p");
histogram!(q_samples[2, :], alpha = 0.7, label = "q");

title!(raw"$\mu_2$")

plot(p1, p2)
