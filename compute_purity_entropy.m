n_c = length(C_list);

pure_SP = zeros(n_c,it_ex);
entr_SP = zeros(n_c,it_ex);
for idx_C = 1:n_c
    for it_ex = 1:n_exectuion
        class_assigned_svd = all_class_assigned{idx_C,it_ex};
        cm = confusionmat(Ytest,class_assigned_svd);
        pure_SP(idx_C,it_ex) = sum(diag(cm))/Ntest;
        a = cm ./ repmat(sum(cm,1),[n_class 1]);
        entr_SP(idx_C,it_ex) = sum((sum(-(a .* log2(a)),1,'omitNan') .* sum(cm,1)))./ Ntest;
    end
end
%% to latex
pure_SP = 100*pure_SP;
pure_SVD = 100*pure_SVD;
avg_pure_SP = mean(pure_SP,2);
avg_pure_SVD = mean(pure_SVD,2);
std_pure_SP = std(pure_SP,0,2);
std_pure_SVD = std(pure_SVD,0,2);

entr_SP = 100*entr_SP;
entr_SVD = 100*entr_SVD;
avg_entr_SP = mean(entr_SP,2);
avg_entr_SVD = mean(entr_SVD,2);
std_entr_SP = std(entr_SP,0,2);
std_entr_SVD = std(entr_SVD,0,2);

for i=1:5
    fprintf(1,"$C=%d$ & $%.2f\\;(%.2f)$ & $%.2f\\;(%.2f)$ & $%.2f\\;(%.2f)$ & $%.2f\\;(%.2f)$\\\\\n",2*i,avg_pure_SP(i), std_pure_SP(i), avg_pure_SVD(i), std_pure_SVD(i),avg_entr_SP(i), std_entr_SP(i), avg_entr_SVD(i), std_entr_SVD(i));
end