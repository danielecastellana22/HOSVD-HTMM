classdef treeRep_s
    %TREEREP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        %ch_list
        ch_list;
        
        %pos list
        pos_list;
        
        %number of nodes
        n;
        
        %number of leaves
        nf;
        
        %visible variable emitted by the nodes
        v;
       
    end
    
    methods
        
        % generate treeRep_s starting from Davide representation
        function obj = treeRep_s(obs,child_mat)
                       
            obj.n = length(obs);
            obj.nf=sum(sum(child_mat,2)==0);
            obj.v = obs;
            
            L_ = size(child_mat,2);
            n_ = obj.n; 
            obj.pos_list = zeros(1,n_);
            for u=1:n_
                for l=1:L_
                    ch_l_u = child_mat(u,l);
                    if(ch_l_u ~=0)
                        obj.pos_list(ch_l_u) = l;
                    end
                end
            end
            
            obj.ch_list = child_mat;
            %change 0 to -1
            obj.ch_list(obj.ch_list==0) = -1;
            
        end
        
        function plot(t,vocabulary)
            dNode = 1000;
            if(nargin < 2)
                vocabulary = [];
            end
            parents = zeros(t.n,1);
            for u=1:t.n
                for l=1:size(t.ch_list,2)
                    ch_id = t.ch_list(u,l);
                    if(ch_id >0)
                        parents(ch_id) = u;
                    end
                end
            end
            
            [x,y] = treelayout(parents);
            hold on;
            
            %draw edges
            for i=1:t.n-t.nf
                for ch = t.ch_list(i,:)
                    if(ch>0)
                        %draw edge
                        line([x(i) x(ch)],[y(i) y(ch)],'Color','k');
                    end
                end
            end
            
            %draw nodes
            n_ch = sum(t.ch_list~=-1,2);
            err = (t.v ~=n_ch);
            if(any(err))
                scatter(x(err),y(err),dNode,'filled','MarkerFaceColor','w','MarkerEdgeColor','r','LineWidth',1.5);
            end
            if(any(~err))
                scatter(x(~err),y(~err),dNode,'filled','MarkerFaceColor','w','MarkerEdgeColor','k','LineWidth',1);
            end
            

            %draw label
            for i=1:t.n
                if(isempty(vocabulary))
                    s = num2str(t.v(i));
                else
                    s = vocabulary(t.v(i));
                end
                text(x(i),y(i),s,'HorizontalAlignment','center','VerticalAlignment','middle')
                if(i~=1)
                    pos = t.pos_list(i);
                else
                    pos = 0;
                end
                %text(x(i)-0.03,y(i)+0.03,num2str(pos),'HorizontalAlignment','center','VerticalAlignment','middle')
            end
            axis off;
        end
        
        function ch_list = get_children_list(t,u)
            ch_list = t.ch_list(u,:);
        end
        
        function pos_u = get_position(t,u)
            pos_u = t.pos_list(u);
        end
        
        function ris = is_leaf(t,u)
            ris = (u > t.n - t.nf);
        end
    end
    
end