function Davide_to_my_representation(dataset_filename,outname)
    load(dataset_filename);
    
    N=length(classtot);
    Ntest = round(N /10);
    Ntr = N- Ntest;
    
    Xtr = cell(Ntr,1);
    Xtest = cell(Ntest,1);
    
    L = size(Ytot.child{1},2);
    M = 0;
    pp = randperm(N);
    
    for i=1:Ntr
        idx = pp(i);
        Xtr{i} = treeRep_s(Ytot.obs{idx},Ytot.child{idx});
        M=max(M,max(Ytot.obs{idx}));
    end
    Ytr = classtot(pp(1:Ntr))';
    
    for i=1:Ntest
        idx = pp(Ntr+i);
        Xtest{i} = treeRep_s(Ytot.obs{idx},Ytot.child{idx});
        M=max(M,max(Ytot.obs{idx}));
    end
    Ytest = classtot(pp(Ntr+1:end))';
    
    save(['datasets/' outname], 'L','M', 'Ntest','Ntr', 'Xtest', 'Xtr', 'Ytest', 'Ytr');
end

