function [pure,entr] = compute_purity_entropy(C_list,n_ex,Ytest,Y_predicted)

%Y_predicted has size [n_c,n_ex]
n_c = length(C_list);

pure = zeros(n_c,n_ex);
entr = zeros(n_c,n_ex);
for idx_C = 1:n_c
    for n_ex = 1:n_exectuion
        cm = confusionmat(Ytest,Y_predicted{idx_C,n_ex});
        pure(idx_C,n_ex) = sum(diag(cm))/Ntest;
        a = cm ./ repmat(sum(cm,1),[n_class 1]);
        entr(idx_C,n_ex) = sum((sum(-(a .* log2(a)),1,'omitNan') .* sum(cm,1)))./ Ntest;
    end
end
% %% to latex
% pure = 100*pure;
% pure = 100*pure;
% avg_pure = mean(pure,2);
% avg_pure = mean(pure,2);
% std_pure = std(pure,0,2);
% std_pure = std(pure,0,2);
% 
% for i=1:5
%     fprintf(1,"$C=%d$ & $%.2f\\;(%.2f)$ & $%.2f\\;(%.2f)$ & $%.2f\\;(%.2f)$ & $%.2f\\;(%.2f)$\\\\\n",2*i,avg_pure(i), std_pure(i), avg_pure(i), std_pure(i),avg_entr_SP(i), std_entr_SP(i), avg_entr_SVD(i), std_entr_SVD(i));
% end
end