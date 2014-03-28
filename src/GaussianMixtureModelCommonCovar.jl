immutable GaussianMixtureModelCommonCovar <: Distribution
    components::Vector # Vector should be able to contain any type of
                       # distribution with comparable support
	mu::Array{Float64,2}
	probs::Vector{Float64}
	invcovar::Array{Float64,2}
    aliastable::AliasTable
    function GaussianMixtureModelCommonCovar(mu::Array{Float64,2}, p::Vector{Float64}, ic::Array{Float64,2})
        if size(mu,2) != length(p)
            error("means and probs must have the same number of elements")
        end
		if( size(invcovar,1) != size(invcovar,2) )
		    error("covariance matrix must be square")
		end 
		if( size(mu,1) != size(invcovar,1) )
		    error("means and covar matrix not compatible size")
		end
        sump = 0.0
        for i in 1:length(p)
            if p[i] < 0.0
                error("MixtureModel: probabilities must be non-negative")
            end
            sump += p[i]
        end
        table = AliasTable(p)
        new(mu, p ./ sump, ic, table)
    end
end

function mean(d::GaussianMixtureModelCommonCovar)
    np = size(d.mu,2)
	m = zeros(np)
    for i in 1:length(d.probs)
        m += squeeze(d.mu[:,i),1) .* d.probs[i]
    end
    return m
end

function pdf(d::GaussianMixtureModelCommonCovar, x::Any)
    p = 0.0
    for i in 1:length(d.probs)
        p += pdf(MvNormalCanon(d.invcov), x .- squeeze(d.mu[:,i],1) ) * d.probs[i]
    end
    return p
end

function rand(d::GaussianMixtureModelCommonCovar)
    i = rand(d.aliastable)
    return squeeze(d.mu[:,i],1) .+ rand(MvNormalCanon(d.invcov))
end

