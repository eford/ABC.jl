if !isdefined(:Distributions) using Distributions end
if !isdefined(:PDMats)        using PDMats end

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

function rand(d::GaussianMixtureModelCommonCovar)
    i = Distributions.rand(d.aliastable)
    return  Distributions.rand(Distributions.MvNormal(vec(d.mu[:,i]),d.covar))
end


