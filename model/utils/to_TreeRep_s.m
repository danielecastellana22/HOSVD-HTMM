function [new_cell_dataset] = to_TreeRep_s(cell_dataset, L)
    n=length(cell_dataset);
    new_cell_dataset = cell(n,1);
    
    for i=1:n
        %new_cell_dataset{i} = treeRep_s(cell_dataset{i},L);
        old_t = cell_dataset{i};

        n_ = old_t.n; 

        ch_list = zeros(n_,L);

        for u=1:n_

            [~,ch_id,pos_id] =  find(old_t.adj(u,:));
            ch_list(u,pos_id) = ch_id;
        end
        
        new_cell_dataset{i} = treeRep_s(old_t.v, ch_list);
    end
end

