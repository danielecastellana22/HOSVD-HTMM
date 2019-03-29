p = genpath('model');
addpath(p);

load('datasets/symm.mat');
M=4;
C=10;
n_it = 100; 
t_0 = 10;
m_0 = 20;
min_lag = 1;
max_lag = 3;
loglike_it_tf = cell(5,1);
loglike_it_sp = cell(5,1);
Y_traj_sp_lf = cell(5,1);
Y_traj_tf_lf = cell(5,1);
err_tf = cell(5,1);
err_sp = cell(5,1);
model_tf = cell(5,1);
model_sp = cell(5,1);
for it=1:5
    m_tf = TF_BHTMM(C, L, M, min_lag, max_lag, 1, 1, ones(1,L), 2, 1, 1);
    [loglike_it_tf{it}, ~,~] = m_tf.train(Xtr,n_it, t_0, m_0,1,0);
    m_sp = SP_BHTMM(C,L,M,1,1,1,1);
    [loglike_it_sp{it},~,~] = m_sp.train(Xtr,n_it,1);
    model_tf{it} = m_tf;
    model_sp{it} = m_sp;
    %% test
    t_0 = 10;
    m_0 = 20;
    myTest_onlyLeaf = change_label(Xtest);
    [Y_traj_sp_lf_aux] = m_sp.labelling(myTest_onlyLeaf,0);
    [loglike_tf_test_lf, Q_traj_tf_lf, ~, ~] = m_tf.test(myTest_onlyLeaf,20, t_0, m_0);


    Y_traj_tf_lf_aux = cell(Ntest,1);
    for i=1:Ntest
        Y_traj_tf_lf_aux{i} = m_tf.sample_visible(myTest_onlyLeaf{i},Q_traj_tf_lf{i});
    end
    Y_traj_tf_lf{it} = Y_traj_tf_lf_aux;
    Y_traj_sp_lf{it} = Y_traj_sp_lf_aux;
    %% stats
    err_tf_aux = zeros(M,M);
    err_sp_aux = zeros(M,M);

    for i=1:Ntest
        for u=1:Xtest{i}.n
            a = Y_traj_tf_lf_aux{i};
            b = Y_traj_sp_lf_aux{i}';
            err_tf_aux(Xtest{i}.v(u),a(u)) = err_tf_aux(Xtest{i}.v(u),a(u)) + 1;
            err_sp_aux(Xtest{i}.v(u),b(u)) = err_sp_aux(Xtest{i}.v(u),b(u)) + 1;
        end
    end
    
    err_sp{it} = err_sp_aux;
    err_tf{it} = err_tf_aux;
end
 
save('results\labelling\symm_SP_TF');

%% compute entropy and purity
pur_sp = zeros(M+1);
pur_tf = zeros(M+1);

entr_sp = zeros(M+1);
entr_tf = zeros(M+1);
for it=1:5
    a_sp = err_sp{it};
    a_tf = err_tf{it};
    pur_sp(1:4,it) = diag(a_sp) ./ sum(a_sp,2);
    pur_tf(1:4,it) = diag(a_tf) ./ sum(a_tf,2);
    pur_sp(5,it) = sum(diag(a_sp)) ./ sum(sum(a_sp));
    pur_tf(5,it) = sum(diag(a_tf)) ./ sum(sum(a_tf));
    
    a = a_sp ./ repmat(sum(a_sp,1),[M 1]);
    entr_sp(1:4,it) = sum(-(a .* log2(a)),1,'omitNan');
    entr_sp(end,it) = sum((sum(-(a .* log2(a)),1,'omitNan') .* sum(a_sp,1)))./ sum(sum(a_sp));
    
    a = a_tf ./ repmat(sum(a_tf,1),[M 1]);
    entr_tf(1:4,it) = sum(-(a .* log2(a)),1,'omitNan');
    entr_tf(end,it) = sum((sum(-(a .* log2(a)),1,'omitNan') .* sum(a_tf,1)))./ sum(sum(a_tf));
end

pur_sp = pur_sp * 100;
pur_tf = pur_tf * 100;
entr_sp = entr_sp * 100;
entr_tf = entr_tf * 100;

acc_sp_mean = mean(pur_sp,2);
acc_tf_mean = mean(pur_tf,2);
acc_sp_std = std(pur_sp,0,2);
acc_tf_std = std(pur_tf,0,2);

entr_sp_mean = mean(entr_sp,2);
entr_tf_mean = mean(entr_tf,2);
entr_sp_std = std(entr_sp,0,2);
entr_tf_std = std(entr_tf,0,2);

%% print table
for i=1:5
     fprintf(1,"$%d$ & $%.2f\\;(%.2f)$ & $%.2f\\;(%.2f)$ & $%.2f\\;(%.2f)$ & $%.2f\\;(%.2f)$\\\\\n",i-1,acc_sp_mean(i),acc_sp_std(i), acc_tf_mean(i), acc_tf_std(i),entr_sp_mean(i),entr_sp_std(i), entr_tf_mean(i), entr_tf_std(i));
end

%% plot confusion matrix best model

% idx best model can changes in different execution
Y_traj_tf_best = Y_traj_tf_lf{2};
Y_traj_sp_best = Y_traj_sp_lf{1};

trueY = zeros(4,10065);
bestTF = zeros(4,10065);
bestSP = zeros(4,10065);

cont=1;
for i=1:Ntest
    for u=1:Xtest{i}.n
        a = Y_traj_tf_best{i};
        b = Y_traj_sp_best{i}';
        bestTF(a(u),cont) = 1;
        bestSP(b(u),cont) = 1;
        trueY(Xtest{i}.v(u),cont) = 1;
        cont=cont+1;
    end
end

f=figure; pp= plotconfusion(trueY,bestSP,'Best SP-BHTMM');
pp.Children(2).XTickLabel = {'0' '1' '2' '3' ''};
pp.Children(2).YTickLabel = {'0' '1' '2' '3' ''};
set(findall(f,'-property','FontSize'),'FontSize',20);
f.PaperUnits = 'inches';
f.PaperPosition = [0 0 10 10];
print(['plots/bets_cm_SP'],'-dpng','-r0');

f=figure; pp= plotconfusion(trueY,bestTF,'Best TF-BHTMM');
pp.Children(2).XTickLabel = {'0' '1' '2' '3' ''};
pp.Children(2).YTickLabel = {'0' '1' '2' '3' ''};
set(findall(f,'-property','FontSize'),'FontSize',20);
f.PaperUnits = 'inches';
f.PaperPosition = [0 0 10 10];
print(['plots/bets_cm_TF'],'-dpng','-r0');

%% plots tree from test set
% idx best model can changes in different execution
best_Y_sp = Y_traj_sp_lf{1};
best_Y_tf = Y_traj_tf_lf{2};

idx=10;

f = figure; a = Xtest{idx}; a.v = best_Y_sp{idx}-1; a.plot(); 
set(findall(f,'-property','FontSize'),'FontSize',20);
f.PaperUnits = 'inches';
f.PaperPosition = [0 0 10 10];
print(['plots/treeSP' num2str(idx)],'-dpng','-r0');


f = figure; a = Xtest{idx}; a.v = best_Y_tf{idx}'-1; a.plot(); 
set(findall(f,'-property','FontSize'),'FontSize',20);
f.PaperUnits = 'inches';
f.PaperPosition = [0 0 10 10];
print(['plots/treeTF' num2str(idx)],'-dpng','-r0');