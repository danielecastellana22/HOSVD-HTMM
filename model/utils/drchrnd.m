function r = drchrnd(a,n)
    if(sum(a)==0)
        a = a + eps;
    end
    p = length(a);
    r = gamrnd(repmat(a,n,1),1,n,p);
    r = r ./ repmat(sum(r,2),1,p);
end