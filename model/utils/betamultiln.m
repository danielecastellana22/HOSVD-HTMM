function ris = betamultiln(alphas)
%BETAMULLN Summary of this function goes here
%   Detailed explanation goes here
    ris = sum(gammaln(alphas)) - gammaln(sum(alphas));
end

