classdef TF_BHTMM < handle
    %INPUTMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        %number of maximum output degree
        L;
        %number of hidden values
        C;
        %number of visibile values
        M;
        
        bottomCh;
        
        % parameters to sample and represents tabs
        alpha_zero; % input
        lamb_zero;  % ~ DIR(alpha_zero/C,...,alpha_zero/C) 
        alpha_lamb; % input
        lamb_tab; % ~ DIR(alpha_lamb * lamb_zero(1), ... , alpha_lamb * lamb_zero(C))
        gamma_lag;  % input (L values)
        phi_exp;    % input
        pi_tab;     % ~ DIR(gamma_lag(j),...,gamma_lag(j))
        k_pr_tab;   % ~ exp(-fi_exp * j * k)
        k_values;   % the values k_j
        alpha_pr;
        alpha_em;             
        pr_tab;
        em_tab;
        
        min_lag;
        max_lag;
        
        % vector: one value for each tree
        Q_traj;
        Z_traj;
        loglike_trees;
    end
    
    methods
        
        function obj = TF_BHTMM(C_,L_,M_, min_lag_, max_lag_, alpha_zero_,alpha_lamb_,gamma_lag_,phi_exp_,alpha_pr_,alpha_em_)           
            obj.C = C_+1;
            obj.bottomCh = obj.C;
            obj.L = L_;
            obj.M = M_;
            
            % alphas for dirichlet priori
            obj.alpha_zero = alpha_zero_;
            obj.alpha_lamb = alpha_lamb_;
            obj.gamma_lag = gamma_lag_;
            obj.phi_exp = phi_exp_;
            obj.alpha_pr = alpha_pr_;
            obj.alpha_em = alpha_em_;
                     
            obj.min_lag = min_lag_;
            obj.max_lag = max_lag_;
            
            %init the potentials
            obj.init_potentials_from_priori();
        end
        
        function [likelihood_vect, time_vect, K_vect] = train(obj,X_list,max_it, t_0, m_0, print_stdout, plot_likelihood)
            
            likelihood_vect = zeros(max_it,1);
            time_vect = zeros(max_it,1);
            K_vect = zeros(max_it,obj.L);
            
            if(nargin<6)
                print_stdout=0;
            end
            
            if(nargin<7)
                plot_likelihood = 0;
            end
            
            if(print_stdout)
                fprintf(1,'Train\n');
            end
                        
            obj.initialise_before_inference(X_list);
            
            for it=1:max_it
                current_temp = max(t_0^(1-it/m_0),1);
                if(print_stdout)
                    fprintf(1,"T=%.2f\n",current_temp);
                end
                
                tic;
                obj.one_step_inference(X_list, current_temp, false, print_stdout);
                t = toc;
                
                loglike_ds = sum(obj.loglike_trees);
                % print on screen
                if(print_stdout)
                    fprintf(1,'It. %3d\tLoglike: %.10f\n',it,loglike_ds);
                end 
                
                likelihood_vect(it) = loglike_ds;
                time_vect(it) = t;
                K_vect(it,:) = obj.k_values;
                
                if plot_likelihood
                    if(it==1)
                        f = figure;
                        p = semilogy(loglike_ds);
                        p.Marker = 'x';
                    else
                        p.YData = [p.YData loglike_ds];
                    end
                    drawnow;
                end
            end
        end
        
        function [likelihood_vect, Q_vect, time_vect, K_vect] = test(obj,X_list,max_it,t_0,m_0,print_stdout)
            
            time_vect = zeros(max_it,1);
            K_vect = zeros(max_it,1);
            
            if(nargin<6)
                print_stdout=0;
            end
            
            if(print_stdout)
                fprintf(1,'Test\n');
            end
            
            obj.initialise_before_inference(X_list);
            
            for it=1:max_it
                current_temp = max(t_0^(1-it/m_0),1);
                if(print_stdout)
                    fprintf(1,"T=%.2f\n",current_temp);
                end
                
                tic;
                obj.one_step_inference(X_list, current_temp, true, print_stdout);
                t = toc;
                
                loglike_ds = sum(obj.loglike_trees);
                % print on screen
                if(print_stdout)
                    fprintf(1,'It. %3d\tLoglike: %.10f\n',it,loglike_ds);
                end
                
                time_vect(it) = t;
                K_vect(it) = obj.C;
            end
            
            % during test, takes the last likelihood computed
            likelihood_vect = obj.loglike_trees;
            Q_vect = obj.Q_traj;
        end
                        
        function one_step_inference(obj,X_list, current_temp, is_test, print_stdout)
        
            % size of the dataset
            N = length(X_list);
            
            n_scart = 0;
            if(print_stdout)
                textprogressbar('Data:');
            end
            for j=1:N
                if(print_stdout)
                    textprogressbar(100*(j/N));
                end
                [obj.Q_traj{j}, obj.Z_traj{j}, obj.loglike_trees(j), scart] = obj.sample_latent_Q_and_Z(X_list{j}, obj.Q_traj{j}, obj.Z_traj{j}, obj.loglike_trees(j), current_temp);
                n_scart = n_scart + scart;
            end
            if(print_stdout)
                textprogressbar('Done'); 
            end
            
            if(~is_test)      
                fprintf(1,'N scart: %d\n',n_scart);
                obj.sample_values(X_list, current_temp);
            end
        end
        
        function [Q_new, Z_new, loglike_new, scart] = sample_latent_Q_and_Z(obj, tree_obj, Q_old, Z_old, loglike_old, current_temp)
            
            L_ = obj.L;
            C_ = obj.C;
            
            n = tree_obj.n;
            nf = tree_obj.nf;
            
            % table for new Q
            Q_new = zeros(1, n);
            
            % table for new Z
            Z_new = zeros(n-nf, L_);
            
            % values to compute acceptance rate. USE LOG
            zn_cn = 0;
            zn_c = 0;
            z_c = 0;
            z_cn = 0;
            
            % compute loglile tree
            loglike_new = 0;
            
            %%%%%%%%%%%% SAMPLE Q %%%%%%%%%%%
            % sample Q state for leaves
            for u=n:-1:n-nf+1                
                y_u = tree_obj.v(u);
                pos_u = tree_obj.get_position(u);
                
                if y_u ~= -1
                    prob = obj.pr_tab(pos_u,:) .* obj.em_tab(:,y_u)';
                else
                    % exclude bottom
                    prob = obj.pr_tab(pos_u,1:end-1);
                end
                
                Q_new(u) = catrnd(prob, 1);
                loglike_new = loglike_new + log(prob(Q_new(u)));
            end
            
            %  sample Q and Z state for internal nodes
            for u=n-nf:-1:1
                y_u = tree_obj.v(u);
                ch_list = tree_obj.get_children_list(u);
                
                idx_lambda_old = cell(1,L_+1);
                idx_lambda_old{end} = ':';
                
                idx_lambda_new = cell(1,L_+1);
                idx_lambda_new{end} = ':';
                
                for l=1:L_
                    idx_lambda_old{l} = Z_old(u,l);
                    % to sample Z_new
                    if(ch_list(l)~=-1)
                        ch_state = Q_new(ch_list(l));
                    else
                        ch_state = obj.bottomCh;
                        loglike_new = loglike_new + log(obj.pr_tab(l,ch_state));
                    end
                    
                    Z_new(u,l) = catrnd(reshape(obj.pi_tab(l,ch_state,:),[1 C_]),1);
                    idx_lambda_new{l} = Z_new(u,l);
                end
                
                lamb_z = reshape(obj.lamb_tab(idx_lambda_old{:}),[1 C_]);
                lamb_zn = reshape(obj.lamb_tab(idx_lambda_new{:}),[1 C_]);
                
                if y_u ~= -1
                    prob = lamb_z .* obj.em_tab(:,y_u)';
                else
                    prob = lamb_z;
                end
                Q_new(u) = catrnd(prob, 1);
                
                loglike_new = loglike_new + log(prob(Q_new(u)));
                
                z_c = z_c + log(lamb_z(Q_old(u)));
                z_cn = z_cn + log(lamb_z(Q_new(u)));
                zn_c = zn_c + log(lamb_zn(Q_old(u)));
                zn_cn = zn_cn + log(lamb_zn(Q_new(u)));
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            acc_value = zn_cn + zn_c - z_c - z_cn;
            acc_value = 1/current_temp * acc_value;
            acc_value = min(exp(acc_value),1);
            p = catrnd([acc_value 1-acc_value],1);
            scart = 0;
            if p==2
                Q_new = Q_old;
                Z_new = Z_old;
                scart = 1;
                loglike_new = loglike_old;
            end
        end
        
        function sample_values(obj, Xlist, current_temp)
            L_ = obj.L;
            C_ = obj.C;
            M_ = obj.M;
            
            Z_traj_ = obj.Z_traj;
            Q_traj_ = obj.Q_traj;
            N = length(Z_traj_);

            min_lag_ = obj.min_lag;
            max_lag_ = obj.max_lag;
            
            % proposed k_values
            new_k_values = obj.k_values;
            old_k_values = obj.k_values;
            new_pi_tab = obj.pi_tab;
            
            %change only one component
            l = randi(L_);

            % decide the new value of k_l
            old_k_val = old_k_values(l);
            if(old_k_val==1)
                new_k_val = 2;
            elseif(old_k_val==C_)
                new_k_val = C_-1;
            else
                p = catrnd([0.5 0.5],1);
                if p==1
                    new_k_val = old_k_val+1;
                else
                    new_k_val = old_k_val-1;
                end
            end
            new_k_values(l) = new_k_val;
            
            n_important_lags = sum(new_k_values > 1);
            
            if(n_important_lags > max_lag_)
                % we should remove one important lag
                prob_remove_lag = ones(1,L_);
                % try to preserve the new lag
                prob_remove_lag(l) = 0;
                prob_remove_lag(new_k_values==1) = 0;
                to_remove = catrnd(prob_remove_lag,1);
                new_k_values(to_remove) = new_k_values(to_remove)-1;
                if(new_k_values(to_remove)>1)
                    % the lag is still important, we have to remove the new
                    % update
                    new_k_values(l) = 1;
                end
            elseif(n_important_lags < min_lag_)
                % we must add another important lag
                prob_new_lag = ones(1,L_);
                prob_new_lag(new_k_values>1) = 0;
                to_incr = catrnd(prob_new_lag, 1);
                new_k_values(to_incr) = 2;
            end
            
            n_important_lags = sum(new_k_values > 1);
            
            assert(n_important_lags >= min_lag_ && n_important_lags<=max_lag_);
            
            log_p_old_k_values = 0;
            log_p_new_k_values = 0;
            %update the cluster
            for l=1:L_
                new_k_val = new_k_values(l);
                old_k_val = obj.k_values(l);
                
                assert(abs(new_k_val-old_k_val)<=1);
                
                %compute priori
                log_p_new_k_values = log_p_new_k_values + log(obj.k_pr_tab(new_k_val));
                log_p_old_k_values = log_p_old_k_values + log(obj.k_pr_tab(old_k_val));
                %modify cluster
                if(new_k_val > old_k_val)
                    % increase move
                    % select cluster with at least 2 state
                    prob = reshape(sum(obj.pi_tab(l,:,:),2),1,[]) > 1;
                    to_split = catrnd(prob, 1);
                    % elements that should be splitted in two cluster
                    el_in_cluster = obj.pi_tab(l,:,to_split);
                    n_el = sum(el_in_cluster);
                    
                    assert(n_el>1);
                    
                    % select number of element in the new class
                    n_new_el = randi(n_el-1);
                    
                    el_new_cluster = false(1,C_);
                    pos = find(el_in_cluster);
                    rperm = randperm(n_el);
                    el_new_cluster(pos(rperm(1:n_new_el))) = 1;

                    new_pi_tab(l,el_new_cluster, to_split) = 0;
                    new_pi_tab(l,el_new_cluster, new_k_val) = 1;
                elseif(new_k_val < old_k_val)
                    % decrease move
                    to_merge1 = catrnd(ones(1,old_k_val), 1);
                    prob = ones(1,old_k_val);
                    prob(to_merge1) = 0;
                    to_merge2 = catrnd(prob, 1);
                    %merge2 in merge1
                    new_pi_tab(l,:,to_merge1) = new_pi_tab(l,:,to_merge1) + new_pi_tab(l,:,to_merge2);
                    % ensure there are no empty states
                    if(to_merge2 < old_k_val)
                        new_pi_tab(l,:,to_merge2) = new_pi_tab(l,:,old_k_val);
                    end
                    new_pi_tab(l,:,old_k_val) = 0;
                end             
            end
            
%             fprintf(1,'Old vals: [');
%             for a=obj.k_values
%                 fprintf(1,'%d ',a);
%             end
%             fprintf(1,']\n');
%             
%             fprintf(1,'New vals: [');
%             for a=new_k_values
%                 fprintf(1,'%d ',a);
%             end
%             fprintf(1,']\n');
            
            lk_old_stats = zeros([obj.k_values C_]); 
            lk_new_stats = zeros([new_k_values C_]);
            
            % table to update the priori and emission table
            em_stats = zeros([C_ M_]);
            pr_stats = zeros([L_ C_]);
            
            new_Z_traj = cell(N,1);
                
            % apply the moves to each tree
            for i=1:N    
                tree_obj = Xlist{i};
                n = tree_obj.n;
                nf = tree_obj.nf;
                z_old = Z_traj_{i};
                z_tmp = Z_traj_{i};
                q_tmp = Q_traj_{i};

                %  on internal nodes
                for u=n-nf:-1:1
                    ch_list = tree_obj.get_children_list(u);
                    for l=1:L_
                        ch_id = ch_list(l);
                        if(ch_id ~= -1)
                            ch_state = q_tmp(ch_id);
                        else
                            ch_state = obj.bottomCh;
                             %consider bottom child
                            pr_stats(l, ch_state) = pr_stats(l, ch_state) + 1;
                        end
                        z_tmp(u,l) = catrnd(reshape(new_pi_tab(l,ch_state,:),[1 C_]),1);
                    end
                    
                    idx_stats = num2cell(z_tmp(u,:));
                    idx_stats{end+1} = q_tmp(u);
                    lk_new_stats(idx_stats{:})=  lk_new_stats(idx_stats{:})+1;

                    idx_stats = num2cell(z_old(u,:));
                    idx_stats{end+1} = q_tmp(u);
                    lk_old_stats(idx_stats{:})=  lk_old_stats(idx_stats{:})+1;
                    
                    if tree_obj.v(u) ~= -1
                        em_stats(q_tmp(u),tree_obj.v(u)) = em_stats(q_tmp(u),tree_obj.v(u))+1;
                    end
                end

                % on the leaves
                for u=n:-1:n-nf+1     
                    pos_u = tree_obj.get_position(u);
                    
                    if tree_obj.v(u) ~= -1
                        em_stats(q_tmp(u),tree_obj.v(u)) = em_stats(q_tmp(u),tree_obj.v(u))+1;
                    end
                    
                    pr_stats(pos_u, q_tmp(u)) = pr_stats(pos_u, q_tmp(u)) + 1;
                end
                
                new_Z_traj{i} = z_tmp;
            end

            % decide if accept the move
            rr_new = reshape(lk_new_stats,[],C_);
            sz = size(rr_new);
            marg_lk_new = 0;
            for i=1:sz(1)
                marg_lk_new = marg_lk_new + betamultiln(obj.alpha_lamb*obj.lamb_zero+rr_new(i,:)) - betamultiln(obj.alpha_lamb*obj.lamb_zero);
            end

            rr_old = reshape(lk_old_stats,[],C_);
            sz = size(rr_old);
            marg_lk_old = 0;
            for i=1:sz(1)
                marg_lk_old = marg_lk_old + betamultiln(obj.alpha_lamb*obj.lamb_zero+rr_old(i,:)) - betamultiln(obj.alpha_lamb*obj.lamb_zero);
            end

            acc_value = (marg_lk_new - marg_lk_old);
            acc_value = acc_value + (log_p_new_k_values - log_p_old_k_values);
            acc_value = 1/current_temp * acc_value;
            acc_value = min(exp(acc_value),1);

            save_moves = catrnd([acc_value 1-acc_value],1);
            lk_stats = lk_old_stats;
            if(save_moves == 1)
                obj.k_values = new_k_values;
                obj.Z_traj = new_Z_traj;
                obj.pi_tab = new_pi_tab;
                lk_stats = lk_new_stats;
            end

            % sample lambda and lambda0
            rr_new = reshape(lk_stats,[],C_);
            sz = size(rr_new);
            aux = zeros(sz);
            m0 = zeros(1,C_);
            for i=1:sz(1)
                % sample lambda
                aux(i,:) = drchrnd(obj.alpha_lamb * obj.lamb_zero + rr_new(i,:),1);

                % sample lamda0
                for cc=1:C_
                    for l=1:rr_new(i,cc)
                        v = obj.alpha_lamb * obj.lamb_zero(cc);
                        p = (v)/(l - 1 + v);
                        m0(cc) = m0(cc) + catrnd([1-p p], 1) - 1;
                    end
                end                    
            end
            
            obj.lamb_tab = reshape(aux, [obj.k_values C_]);
            obj.lamb_zero = drchrnd(obj.alpha_zero / C_ + m0,1);

            % sample emission
            for cc=1:C_-1
                obj.em_tab(cc,:) = drchrnd(em_stats(cc,:)+ obj.alpha_em, 1);
            end
            
            % sample priori
            for l=1:L_
                obj.pr_tab(l,:) = drchrnd(obj.alpha_pr + pr_stats(l,:),1);
            end
        end
        
        function v_list = sample_visible(obj, tree_obj, Q_list)
            n = tree_obj.n;
            v_list= zeros(1,n);
            
            for u=1:n
                q_u = Q_list(u);
                prob = obj.em_tab(q_u,:);
                v_u = catrnd(prob,1);
                v_list(u) = v_u;
            end
        end
    end
    
    methods(Access=private)
        
        function init_potentials_from_priori(obj)
            %variables to update potentials
            L_ = obj.L;
            C_ = obj.C;
            M_ = obj.M;
            
            % TODO: read as argument
            min_lag_ = obj.min_lag;
            max_lag_ = obj.max_lag;
            
            % set same starting point
            %rng(151)
            %rng(0)
            %rng('shuffle')
            
            obj.k_pr_tab = exp(-obj.phi_exp * (1:C_));
            obj.k_pr_tab = obj.k_pr_tab ./ sum(obj.k_pr_tab);
            
            obj.k_values = zeros(1,L_);
            pp = randperm(L_);
            % choose min_lag position which will have k > 1
            important_lags = pp(1:min_lag_);
            obj.k_values(important_lags) = catrnd([0 obj.k_pr_tab(2:end)], length(important_lags))';
            % choose L_ - max_lag position which will have k == 1
            not_important_lags = pp(max_lag_+1:end);
            if(~isempty(not_important_lags))
                obj.k_values(not_important_lags) = 1;
            end
            % all the others can be choose randomly
            other_lags = pp(min_lag_+1:max_lag_);
            if(~isempty(other_lags))
                obj.k_values(other_lags) = catrnd(obj.k_pr_tab, length(other_lags))';
            end
            
            %obj.k_values = catrnd(obj.k_pr_tab, L_)';
            obj.lamb_zero = drchrnd((obj.alpha_zero / C_) .* ones(1,C_), 1);
            
            n_comb = prod(obj.k_values);
            obj.lamb_tab = reshape(drchrnd(obj.alpha_zero * obj.lamb_zero, n_comb), [obj.k_values C_]);
            
            obj.pi_tab = zeros([L_ C_ C_]);
            for l=1:obj.L
                %obj.pi_tab(l,:,1:obj.k_values(l)) = drchrnd(obj.gamma_lag(l) * ones(1,obj.k_values(l)), C_);
                for c=1:C_
                    set_to_one = catrnd(ones(1,obj.k_values(l)), 1);
                    obj.pi_tab(l,c,set_to_one) = 1;
                end
            end
            
            obj.pr_tab = drchrnd(obj.alpha_pr * ones(1,C_),L_);
                                                
            obj.em_tab = zeros(obj.C, obj.M);
            obj.em_tab(1:end-1,:) = drchrnd(obj.alpha_em * ones(1,obj.M),obj.C-1);            
        end
        
        function initialise_before_inference(obj, X_list)
            N = length(X_list);
            L_ = obj.L;
            C_ = obj.C;
            %initialise trajectories
            obj.Q_traj = cell(N,1);
            obj.Z_traj = cell(N,1);
            obj.loglike_trees = zeros(N,1);

            for j=1:N
                tree_obj = X_list{j};
                n = tree_obj.n;
                nf = tree_obj.nf;
                n_int = n - nf;
                
                Q_tmp = zeros(1,n);
                Z_tmp = zeros(n_int, L_);   
                loglike_val = 0;
                
                % sample Q state for leaves
                for u=n:-1:n-nf+1                
                    pos_u = tree_obj.get_position(u);   
                    v_u = tree_obj.v(u);
                    
                    if v_u ~= -1
                        prob = obj.pr_tab(pos_u,:) .* obj.em_tab(:,v_u)';
                    else
                        % exclude bottom
                        prob = obj.pr_tab(pos_u,1:end-1);
                    end
                    
                    Q_tmp(u) = catrnd(prob, 1);
                    
                    loglike_val = loglike_val + log(prob(Q_tmp(u)));
                end

                %  sample Q and Z state for internal nodes
                for u=n-nf:-1:1
                    ch_list = tree_obj.get_children_list(u);
                                    
                    idx_lambda = cell(1,L_+1);
                    idx_lambda{end} = ':';
                    for l=1:L_
                        % to sample Z_new
                        if(ch_list(l)~=-1)
                            ch_state = Q_tmp(ch_list(l));
                        else
                            ch_state = obj.bottomCh;
                            loglike_val = loglike_val + log(obj.pr_tab(l,ch_state));
                        end

                        Z_tmp(u,l) = catrnd(reshape(obj.pi_tab(l,ch_state,:),[1 C_]),1);
                        idx_lambda{l} = Z_tmp(u,l);
                    end
                    v_u = tree_obj.v(u);
                    if (v_u  ~= -1)
                        prob = reshape(obj.lamb_tab(idx_lambda{:}),[1 C_]) .* obj.em_tab(:,v_u)';
                    else
                        prob = reshape(obj.lamb_tab(idx_lambda{:}),[1 C_]);
                    end
                    
                    Q_tmp(u) = catrnd(prob, 1) ;
                    
                    loglike_val = loglike_val + log(prob(Q_tmp(u)));
                end
                
                obj.Q_traj{j} = Q_tmp;
                obj.Z_traj{j} = Z_tmp;
                obj.loglike_trees(j) = loglike_val;
            end
        end
        
    end
end

