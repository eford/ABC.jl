if !isdefined(:Distributions) using Distributions end
if !isdefined(:PDMats)        using PDMats end

immutable GaussianMixtureModelCommonCovarSubset <: Distribution
     gmm::GaussianMixtureModelCommonCovar
     param_active::Vector{Int64}
     function GaussianMixtureModelCommonCovarSubset(m::Array{Float64,2}, p::Vector{Float64}, ic::AbstractPDMat, pact::Vector{Int64} )
        @assert(1<=length(pact)<=length(p))
        for idx in pact
            if ! 1<=idx<=length(p)
               error("active parameters not in range 1:",length(p))
            end
        end
        new( GaussianMixtureModelCommonCovar(m,p,ic), copy(pact) )
     end
end


function mean(d::GaussianMixtureModelCommonCovarSubseti, x::Any)
    m = copy(x)
    for i in 1:length(d.probs)
        m[param_active] += vec(d.mu[:,i]) .* d.probs[i]
    end
    return m
end

function pdf(d::GaussianMixtureModelCommonCovar, x::Any)
    p = 0.0
    for i in 1:length(d.probs)
        p += Distributions.pdf(Distributions.MvNormal(d.covar), x .- vec(d.mu[:,i]) ) * d.probs[i]
    end
    return p
end

function logpdf(d::GaussianMixtureModelCommonCovar, x::Any)
    logp = -Inf
    for i in 1:length(d.probs)
        if d.probs[i]<=0.0 continue end
        logp_i = Distributions.logpdf(Distributions.MvNormal(d.covar), x .- vec(d.mu[:,i]) ) + log(d.probs[i])
        logp = logp > logp_i ? logp + log1p(exp(logp_i-logp)) : logp_i + log1p(exp(logp-logp_i))
    end
    return logp
end

# Need to know dimension
function rand(d::GaussianMixtureModelCommonCovar)
    i = Distributions.rand(d.aliastable)
    return  Distributions.rand(Distributions.MvNormal(vec(d.mu[:,i]),d.covar))
end


immutable GaussianMixtureModelCommonCovar <: Distribution
	mu::Array{Float64,2}
	probs::Vector{Float64}
        covar::AbstractPDMat
        aliastable::Distributions.AliasTable

    function GaussianMixtureModelCommonCovar(m::Array{Float64,2}, p::Vector{Float64}, ic::AbstractPDMat)
        if size(m,2) != length(p)
            error("means and probs must have the same number of elements")
        end
		if( size(ic,1) != size(ic,2) )
		    error("covariance matrix must be square")
		end 
		if( size(m,1) != size(ic,1) )
		    error("means and covar matrix not compatible sizes: ",size(m)," vs ",size(ic) )
		end
        sump = 0.0
        for i in 1:length(p)
            if p[i] < 0.0
                error("MixtureModel: probabilities must be non-negative")
            end
            sump += p[i]
        end
        table = Distributions.AliasTable(p)
        new(m, p ./ sump, ic, table)
    end
end

function mean(d::GaussianMixtureModelCommonCovar)
    np = size(d.mu,2)
	m = zeros(np)
    for i in 1:length(d.probs)
        m += vec(d.mu[:,i]) .* d.probs[i]
    end
    return m
end

function pdf(d::GaussianMixtureModelCommonCovar, x::Any)
    p = 0.0
    for i in 1:length(d.probs)
        p += Distributions.pdf(Distributions.MvNormal(d.covar), x .- vec(d.mu[:,i]) ) * d.probs[i]
    end
    return p
end

function logpdf(d::GaussianMixtureModelCommonCovar, x::Any)
    logp = -Inf
    for i in 1:length(d.probs)
        if d.probs[i]<=0.0 continue end
        logp_i = Distributions.logpdf(Distributions.MvNormal(d.covar), x .- vec(d.mu[:,i]) ) + log(d.probs[i])
        logp = logp > logp_i ? logp + log1p(exp(logp_i-logp)) : logp_i + log1p(exp(logp-logp_i))
    end
    return logp
end

function rand(d::GaussianMixtureModelCommonCovar)
    i = Distributions.rand(d.aliastable)
    return  Distributions.rand(Distributions.MvNormal(vec(d.mu[:,i]),d.covar))
end


