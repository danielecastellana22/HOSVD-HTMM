function hd = change_label(d)
    hd = d;
    for i=1:length(d)
        t = hd{i};
        %t.v(1:t.n-t.nf) = -ones(1,t.n-t.nf);
        t.v = -ones(1,t.n);
        %t.v(1) = ceil(3*i/length(d)); 
        hd{i} = t;
    end
end