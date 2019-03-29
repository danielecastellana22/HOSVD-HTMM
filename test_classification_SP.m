function test_classification_SP(dname,C_list,folder_name)
%test_classification_SP Execute classification task on different dataset
%using SP-BHTMM.
%   dname -> the name of the dataset
%   C_list -> the list of different C values to validate
%   folder_name -> folder where to store results
    p = genpath('model');
    addpath(p);
    
    if(strcmp(dname,'symm'))
        fprintf(1,'Loading %s\n',dname);
        load('datasets/symm.mat');
    end
    if(strcmp(dname,'inex05'))
        fprintf(1,'Loading %s\n',dname);
        load('datasets/inex05.mat');
    end
    if(strcmp(dname,'inex06'))
        fprintf(1,'Loading %s\n',dname);
        load('datasets/inex06.mat');
    end 

    n_it = 100; 
    n_exectuion = 5;

    L=L;
    M=M;
    Ytr = Ytr;
    Xtr = Xtr;
    Xtest = Xtest;
    myTr = to_TreeRep_s(Xtr, L);
    myTest = to_TreeRep_s(Xtest, L);
    n_class = max(Ytr);
    
    all_models = cell(length(C_list),n_exectuion);
    all_K_values = cell(length(C_list),n_exectuion);
    all_loglike = cell(length(C_list),n_exectuion);
    all_time_train = cell(length(C_list),n_exectuion);
    all_time_test = cell(length(C_list),n_exectuion);
    all_loglike_test = cell(length(C_list),n_exectuion);
    all_class_assigned = cell(length(C_list),n_exectuion);
    all_class_one_hot_encoded = cell(length(C_list),n_exectuion);
    all_accuracy = zeros(length(C_list),n_exectuion);
    
    %parpool();
   
    for idx_C=1:length(C_list)
        C = C_list(idx_C);
        for it_ex=1:n_exectuion
            savename=['test_' dname '_one_for_class_C' num2str(C) '_it' num2str(it_ex)];
            savepath = ['results/' folder_name '/' savename];
            
            fprintf(1,"C=%d execution %d\n",C, it_ex);
            % first col app, secon sp
            m_cell = cell(n_class,1);
            loglike_cell = cell(n_class,1);
            time_cell = cell(n_class,1);
            K_vals_cell = cell(n_class,1);
            %parfor i=1:n_class    
            for i=1:n_class    
                fprintf(1,"Start training class %d\n",i);
                m_sp = SP_BHTMM(C, L, M, 1,1, 1, 1);

                idx_to_train = (Ytr == i);
                [loglike_it_sp, time_it_sp, k_vect] = m_sp.train(myTr(idx_to_train),n_it,1,0);

                m_cell{i} = m_sp;
                loglike_cell{i} = loglike_it_sp;
                time_cell{i} = time_it_sp;
                K_vals_cell{i} = k_vect;
                fprintf(1,"End training class %d\n",i);
            end
            all_models{idx_C,it_ex} = m_cell;
            all_loglike{idx_C,it_ex} = loglike_cell;
            all_time_train{idx_C,it_ex} = time_cell;
            all_K_values{idx_C,it_ex} = K_vals_cell;
            save(savepath);
            % test
            fprintf(1,"Start test\n");
            n_it_test = 5;
            loglike_it_sp = zeros(Ntest, n_class);
            time_test=zeros(n_it_test,n_class);
            %parfor j=1:n_class
            for j=1:n_class
                m_sp = m_cell{j};
                [loglike_it_sp(:,j), time_test(:,j), ~] = m_sp.test(myTest,n_it_test,1);
            end
            fprintf(1,"End test\n");
            [~, class_assigned_sp] = max(loglike_it_sp,[],2);

            a = zeros(Ntest,n_class);
            b = zeros(Ntest,n_class);

            corr = 0;
            for i=1:Ntest
                a(i,Ytest(i)) = 1;
                b(i,class_assigned_sp(i)) = 1;
                if(Ytest(i) == class_assigned_sp(i))
                    corr = corr+1;
                end
            end
            
            all_time_test{idx_C,it_ex} = time_test;
            all_loglike_test{idx_C,it_ex} = loglike_it_sp;
            all_class_assigned{idx_C,it_ex} = class_assigned_sp;
            all_class_one_hot_encoded{idx_C,it_ex} = b;
            all_accuracy(idx_C,it_ex) = corr / Ntest;
            
            save(savepath);
        end
    end

end