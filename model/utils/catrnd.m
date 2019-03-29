function r = catrnd(w,n)
   
    sumw = sum(w);
    if ~(sumw > 0) || ~all(w>=0) % catches NaNs
        %since everything is zero, chose one randmoly
        w = eps;
    end
    
    r = randsample(length(w),n,true,w);
end

