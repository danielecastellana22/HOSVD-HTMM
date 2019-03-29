classdef SP_BHTMM < handle
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
        
        % alpha dirichlet priori over tables
        alpha_pi;
        alpha_pr;
        alpha_em;
        alpha_sp;
              
        %the potentials
        pr_tab;
        em_tab;
        pi_tab;
        sp_tab;
        
        % variables to accumulate the statistics
        newPrTab;
        newSPTab;
        newPiTab;
        newEmTab;
        
        % variable to accumulate intermediate reuslts
        betaVals;
        
        loglike_trees;
        
    end
    
    methods
        
        function obj = SP_BHTMM(C_,L_,M_,alpha_pi_,alpha_pr_,alpha_em_,alpha_sp_)
            obj.C = C_+1;
            obj.bottomCh = obj.C;
            obj.L = L_;
            obj.M = M_;
            
            % alphas for dirichlet priori
            obj.alpha_pi = alpha_pi_;
            obj.alpha_pr = alpha_pr_;
            obj.alpha_em = alpha_em_;
            obj.alpha_sp = alpha_sp_;
                     
            %init the potentials
            obj.init_potentials_from_priori();
        end
        
        function [likelihood_vect, time_vect, K_vect] = train(obj,X_list,max_it, print_stdout, plot_likelihood)
            
            likelihood_vect = zeros(max_it,1);
            time_vect = zeros(max_it,1);
            K_vect = zeros(max_it,obj.L);
            
            if(nargin<4)
                print_stdout=0;
            end
            
            if(nargin<5)
                plot_likelihood = 0;
            end
            
            if(print_stdout)
                fprintf(1,'Train\n');
            end
                        
            obj.initialise_before_inference(X_list);
            
            for it=1:max_it
                
                tic;
                obj.one_step_inference(X_list, false, print_stdout);
                t = toc;
                
                loglike_ds = sum(obj.loglike_trees);
                % print on screen
                if(print_stdout)
                    fprintf(1,'It. %3d\tLoglike: %.10f\n',it,loglike_ds);
                end 
                
                likelihood_vect(it) = loglike_ds;
                time_vect(it) = t;
                
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
        
        function [likelihood_vect, time_vect, K_vect] = test(obj,X_list,max_it,print_stdout)
            
            time_vect = zeros(max_it,1);
            K_vect = zeros(max_it,1);
            
            if(nargin<4)
                print_stdout=0;
            end
            
            if(print_stdout)
                fprintf(1,'Test\n');
            end
            
            obj.initialise_before_inference(X_list);
            
            for it=1:max_it
                
                tic;
                obj.one_step_inference(X_list, true, print_stdout);
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
        end
        
        function v_cell = labelling(obj, X_list, print_stdout)
            Ntest = length(X_list);
            v_cell = cell(Ntest,1);
            for i=1:Ntest
                v_cell{i} = obj.label_one_tree(X_list{i});
            end
        end
        
        function v_new = label_one_tree(obj, t_obj)
           
            L_ = obj.L;
            C_ = obj.C;
            M_ = obj.M;
        
            bottomCh_ = obj.bottomCh;
            pr_tab_ = obj.pr_tab;
            pi_tab_ = obj.pi_tab;
            em_tab_ = obj.em_tab;
            sp_tab_ = obj.sp_tab;
            
            n = t_obj.n;
            nf = t_obj.nf;
            v = t_obj.v;
            
            v_new = zeros(n,1);
            q_states = zeros(n,1);
            loglike = 0;
            
            %compute betaVals for the leaf
            for u=n-nf+1:n
                l =  t_obj.get_position(u);
                %exclude bottom
                pr = pr_tab_(l,1:end-1);
                
                q_states(u) = catrnd(pr,1);
                
                v_new(u) = catrnd(em_tab_(q_states(u),:),1);
            end
            
            %compute betaVals for the internal nodes
            for u=n-nf:-1:1
                allCh = t_obj.get_children_list(u);
                %table for beta childs
                trans_prob = 0;
                for l=1:L_
                    ch = allCh(l);
                    if(ch~=-1)
                        q_ch = q_states(ch);
                    else
                        q_ch = bottomCh_;
                    end
                    trans_prob = trans_prob + sp_tab_(l)*reshape(pi_tab_(l,q_ch,:),[1 C_]);
                end
                               
                q_states(u) = catrnd(trans_prob,1);
                v_new(u) = catrnd(em_tab_(q_states(u),:),1);
            end
            
        end
        
        function one_step_inference(obj,X_list,is_test, print_stdout)
            n = length(X_list);
            
            % sample the assignment z
            if(print_stdout)
                textprogressbar('Data:');
            end
            for i=1:n
                if(print_stdout)
                    textprogressbar(100*(i/n));
                end
                X_tree = X_list{i};
                lk = obj.upward(X_tree);
                obj.loglike_trees(i) = lk;       
                if(~is_test)
                    obj.downward(X_tree);
                end
            end
            if(print_stdout)
                textprogressbar('Done');
            end
            
            if(~is_test)
                obj.M_step();
            end

        end
        
        function loglike = upward(obj,t_obj)
            %UPWARDINPUT This function executes the upward pass on the input tree X.
                        
            %read variables from obj
            L_ = obj.L;
            C_ = obj.C;
            M_ = obj.M;
        
            bottomCh_ = obj.bottomCh;
            pr_tab_ = obj.pr_tab;
            pi_tab_ = obj.pi_tab;
            em_tab_ = obj.em_tab;
            sp_tab_ = obj.sp_tab;
            
            n = t_obj.n;
            nf = t_obj.nf;
            v = t_obj.v;
            
            %betaVals contains potentials over the h_state_pa variable
            betaVals_ = cell(n,1);
            
            loglike = 0;
            
            %compute betaVals for the leaf
            for u=n-nf+1:n
                l =  t_obj.get_position(u);
                evid = em_tab_(:,v(u));
                pr = pr_tab_(l,:);
                
                beta_tab = evid .* pr';
                
                Nu = sum(beta_tab);
                loglike = loglike + log(Nu);
                
                betaVals_{u} = beta_tab ./ Nu;
            end
            
            %compute betaVals for the internal nodes
            for u=n-nf:-1:1
                allCh = t_obj.get_children_list(u);
                %table for beta childs
                betaChTab = zeros(L_, C_);
                for l=1:L_
                    ch = allCh(l);
                    if(ch~=-1)
                        betaChTab(l,:) = betaVals_{ch};
                    else
                        betaChTab(l,bottomCh_) = 1;
                    end
                end
                               
                evid = em_tab_(:,v(u));
                
                numPot = reshape(sum(sum(pi_tab_ .* repmat(betaChTab, [1 1 C_]), 2) .* repmat(sp_tab_,[1 1 C_]), 1), [C_ 1]);
                
                beta_tab =  evid .* numPot;
                
                Nu = sum(beta_tab);
                loglike = loglike + log(Nu);
                
                betaVals_{u} = beta_tab ./ Nu;
            end
            
            obj.betaVals = betaVals_;
        end
    
        function downward(obj,t_obj)
         	%DOWNARDIN Summary of this function goes here
            %   Detailed explanation goes here
            
            %read variables from obj
            L_ = obj.L;
            C_ = obj.C;
            M_ = obj.M;
            
            bottomCh_ = obj.bottomCh;
            pr_tab_ = obj.pr_tab;
            pi_tab_ = obj.pi_tab;
            em_tab_ = obj.em_tab;
            sp_tab_ = obj.sp_tab;
                        
            betaVals_ = obj.betaVals;
            
            newPrTab_ = obj.newPrTab;
            newPiTab_ = obj.newPiTab;
            newEmTab_ = obj.newEmTab;
            newSPTab_ = obj.newSPTab;
                      
            n = t_obj.n;
            nf = t_obj.nf;
            v = t_obj.v;    
            
            etaVals_ = cell(n,1);
            
            etaVals_{1} = reshape(betaVals_{1}, [1 1 C_]);
            
            %update the b
            newEmTab_(:,v(1)) = newEmTab_(:,v(1)) + betaVals_{1};
            
            %num tab
            sp_pi = repmat(sp_tab_, [1 C_ C_]) .* pi_tab_;
            
            %for each internal node
            for u=1:n-nf
                allCh = t_obj.get_children_list(u);
                betaChTab = -zeros(L_,C_);
                
                for l=1:L_
                    ch = allCh(l);
                    if(ch~=-1)
                        betaChTab(l,:) = betaVals_{ch};
                    else
                        betaChTab(l,bottomCh_) = 1;
                    end
                end
                
                % 1 x 1 x C_
                etaPa = etaVals_{u};
                
                numPot = repmat(betaChTab, [1 1 C_]) .* sp_pi;
                % 1 x 1 x C_
                denPot = sum(sum(numPot,1), 2);
               
                denPot(denPot==0) = eps;
                %P(Q_ch = j, Q_pa = i, Sp = l | X)
                etaChVals_u = (repmat(etaPa, [L_ C_ 1]) .* numPot) ./ repmat(denPot,[L_ C_ 1]);
                
                %P(Q_ch = j, Q_pa = i | Sp = l  X)
                Z = sum(sum(etaChVals_u,2), 3);
                etaChValsCond_u = etaChVals_u ./ repmat(Z,[1 C_ C_]);
                              
                %update A_ 
                % WHICH IS THE CORRECT ONE?--------------------------------
                %newPiTab_ = newPiTab_ + condpot(etaChVals_u,[h_state_ch_ h_state_pa_],[tp_state_ ch_pos_]);
                %newPiTab_ = newPiTab_ + condpot(jointPot,[h_state_ch_ h_state_pa_],[tp_state_ ch_pos_]);
                %newPiTab_ = newPiTab_ + etaChVals_u;
                newPiTab_ = newPiTab_ + etaChValsCond_u;
                %----------------------------------------------------------
                %update SP_
                newSPTab_ = newSPTab_ + Z;
                
                %set eta childs
                %P(Q_ch = j | Sp = l, X)
                etaChPot = sum(etaChValsCond_u, 3);
                for l=1:L_
                    ch = allCh(l);
                    
                    %P(Q_ch = j | Sp = l, X)
                    etaPot = etaChPot(l,:);
                    
                    if(ch~=-1)
                        
                        etaVals_{ch} = reshape(etaPot,[1 1 C_]);
                        
                        %update b_                        
                        newEmTab_(:,v(ch)) = newEmTab_(:,v(ch)) + reshape(etaPot, [C_ 1]);
                        
                        if(ch>n-nf)
                            %is a leaf -> update priori                            
                            newPrTab_(l,:) = newPrTab_(l,:) + etaPot;
                        end
                    else
                        %is a Bottom child! update the prioiri                            
                        newPrTab_(l,:) = newPrTab_(l,:) + etaPot;
                    end
                end
            end
            
            obj.betaVals = [];
            
            obj.newPiTab = newPiTab_;
            obj.newEmTab = newEmTab_;
            obj.newSPTab = newSPTab_;
            obj.newPrTab = newPrTab_;
        end
        
        function M_step(obj)        
            
            L_ = obj.L;
            C_ = obj.C;
            M_ = obj.M;
            
            %M-STEP pr OK
            Z = sum(obj.newPrTab, 2);
            obj.pr_tab = obj.newPrTab ./ repmat(Z, [1 C_]);

            %M-STEP SP OK
            obj.sp_tab = obj.newSPTab ./ sum(obj.newSPTab);            

            %M-STEP pi OK
            Z = sum(obj.newPiTab, 3);
            obj.pi_tab = obj.newPiTab ./ repmat(Z,[1 1 C_]);

            %M-STEP em OK
            Z = sum(obj.newEmTab,2);
            obj.em_tab = obj.newEmTab ./ repmat(Z,[1 M_]);
        end
             
        function init_counting_table(obj)
            %variables to update potentials
            L_ = obj.L;
            C_ = obj.C;
            M_ = obj.M;
           
            %bigEps = 10^-8;
            
            obj.newPrTab = zeros(L_,C_) + (obj.alpha_pr-1) + eps;
            
            obj.newSPTab = zeros(L_,1) + (obj.alpha_sp-1) + eps;
            
            obj.newPiTab = zeros(L_,C_,C_) + (obj.alpha_pi-1) + eps;
            
            obj.newEmTab = zeros(C_,M_) + (obj.alpha_em-1) + eps;            
        end 
        
        % in this case there is no initialisation
        function initialise_before_inference(obj, X_list)
           obj.init_counting_table();
           obj.loglike_trees = zeros(length(X_list),1);
        end
        
        function init_potentials_from_priori(obj)
            %variables to update potentials
            L_ = obj.L;
            C_ = obj.C;
            M_ = obj.M;
            
            alpha_pi_ = obj.alpha_pi;
            alpha_em_ = obj.alpha_em;
            alpha_pr_ = obj.alpha_pr;
            alpha_sp_ = obj.alpha_sp;
            
            % set same starting point
            %rng(0)
            
            obj.pr_tab = drchrnd(alpha_pr_*ones(1,C_),L_);
            
            obj.sp_tab = drchrnd(alpha_sp_*ones(1,L_),1)';
                        
            % pos, ch, pa
            obj.pi_tab = zeros(obj.L, obj.C, obj.C);
            for l=1:L_
                obj.pi_tab(l,:,1:end-1) = drchrnd(alpha_pi_*ones(1,C_-1), C_);
            end
                        
            obj.em_tab = zeros(obj.C, obj.M);
            obj.em_tab(1:end-1,:) = drchrnd(alpha_em_ * ones(1,obj.M),obj.C-1);            
        end
    end
end

