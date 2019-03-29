classdef treeRep
    %TREEREP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        %adjacency matrix
        adj;
        
        %number of nodes
        n;
        
        %number of leaves
        nf;
        
        %visible variable emitted by the nodes
        v;
        
    end
    
    methods(Static)
        %TODO: delete this function since it is task dependent
        %create a tree form a string in bracket notation
        function obj = fromString(s,L,nNodes,charDelim)
                        
            if(nargin<4)
                startTree = '{';
                endTree = '}';
            else
                startTree = charDelim.startTree;
                endTree = charDelim.endTree;
            end
            
            %initialitize attributes
            adj_ = zeros(nNodes,nNodes);
            v_ = zeros(nNodes,1);
            nf_=0;
            newId = 1;
            
            %initialise queue
            queueId = zeros(nNodes,1);
            queuePosCh = zeros(nNodes,1);
            nextPosQ = 1;
            
            %idx to read from s
            readCharPos = 1;
            
            %read type root
            %skip the bracket
            startLabel = readCharPos+1;
            endLabel = startLabel;
            while(s(endLabel) ~= startTree && s(endLabel) ~= endTree)
                endLabel = endLabel+1;
            end
            nodeLabelS = s(startLabel:endLabel-1);
            v_(1) = str2double(nodeLabelS);
            readCharPos = endLabel;
            
            %put the root id in the queue
            queueId(nextPosQ) = 1;
            queuePosCh(nextPosQ) = 1;
            
            %increase newId
            newId = newId + 1;
            
            while(nextPosQ>0)
                %read the id of parent node
                paId = queueId(nextPosQ);
                posCh = queuePosCh(nextPosQ);

                if(s(readCharPos) ~= endTree)
                    %there is a child
                    
                    %there is a bracket, skip it
                    readCharPos = readCharPos +1;
                    
                    %read label
                    startLabel = readCharPos;
                    endLabel = startLabel;
                    while(s(endLabel) ~= startTree && s(endLabel) ~= endTree)
                        endLabel = endLabel+1;
                    end
                    nodeLabelS = s(startLabel:endLabel-1);
                    nodeLabel = str2double(nodeLabelS);
                    readCharPos = endLabel;
                    
                    %update structure
                    v_(newId) = nodeLabel;
                    adj_(paId,newId) = posCh;
                    queuePosCh(nextPosQ) = posCh + 1;
                    
                    if(s(readCharPos) == endTree)
                        %is a leaf
                        readCharPos = readCharPos+1;
                        nf_ = nf_ +1;
                    else
                        %add a stack
                        nextPosQ = nextPosQ +1;
                        queueId(nextPosQ) = newId;
                        queuePosCh(nextPosQ) = 1;
                    end
                    
                    %update newId
                    newId = newId +1;
                else
                    %there are no more child
                    queueId(nextPosQ) = 0;
                    queuePosCh(nextPosQ) = 0;
                    nextPosQ = nextPosQ-1;
                    %shift char
                    readCharPos = readCharPos+1;
                end
            end
            %format the tree
            obj = treeRep(adj_,nNodes,nf_,v_,L);
            obj = obj.format();
        end
    end
    
    methods
        function obj = treeRep(adj,n,nf,v,L)
            obj.adj = adj;
            obj.n = n;
            obj.nf = nf;
            if(iscolumn(v))
                v=v';
            end
            obj.v = v;
        end
        
        %charDelim.startTree indicates the char to indicate the start of a
        %tree
        %charDelim.endTree indicates the char to indicate the end of a
        %tree
        function s = toString(obj,charDelim)
            
            if(nargin<2)
                startTree = '{';
                endTree = '}';
            else
                startTree = charDelim.startTree;
                endTree = charDelim.endTree;
            end
            
            adj_ = obj.adj;
            v_ = obj.v;
            n_ = obj.n;
            L_ = obj.L;
            
            %stack to execute DFS
            % -1 -> fake state to inser }
            %  0 -> bottom state
            % id -> id of the nodes to visit
            stack = zeros(2*n_,1);
            
            s = blanks(10*n_);
            nextC = 1;
            
            stack(1) = 1;
            nextR = 1;
            while(nextR ~=0)
                pa = stack(nextR);
                nextR = nextR - 1;
                
                if(pa>0)
                    %write
                    s(nextC) = startTree;
                    idLabel = num2str(v_(pa));
                    nChar = length(idLabel);
                    s(nextC+1:nextC+nChar) = idLabel;
                    nextC = nextC +nChar +1;
                    
                    %add in the stack a fake state to balance the brackets
                    stack(nextR + 1) = -1;
                    nextR = nextR + 1;
                    
                    
                    %add childs on the stack
                    [~,setCh,posCh] = find(adj_(pa,:));
                    if(~isempty(setCh))
                        %is not a leaf
                        allCh = zeros(L_,1);
                        allCh(posCh) = setCh;
                        for l=L_:-1:1
                            stack(nextR + 1) = allCh(l);
                            nextR = nextR + 1;
                        end
                    end
                else
                    if(pa==-1)
                        %close the brackets
                        s(nextC) = endTree;
                        nextC = nextC+1;
                    end
                    if(pa==0)
                        %generate the bottom child
                        s(nextC) = startTree;
                        s(nextC+1) = '0';
                        s(nextC+2) = endTree;
                        nextC = nextC+3;
                    end
                end
            end
            s = s(1:nextC-1);
        end
        
        %assign the id of the nodes s.t. the id's leaves are greater than
        %id's internal
        function newObj = format(obj)
            n_ = obj.n;
            nf_ = obj.nf;
            adj_ = obj.adj;
            nInt = n_ - nf_;
            
            %list of leaves
            leaves = not(logical(sum(adj_,2)));
            %parent of leaves in the new enumeration
            newPaLeaves = adj_(~leaves,leaves);
            
            intMat = adj_(~leaves,~leaves);
            
            adjMat = zeros(n_);
            adjMat(1:nInt,1:nInt) = intMat;
            adjMat(1:nInt,nInt+1:end) = newPaLeaves;
            
            renameV = zeros(n_,1);
            renameV(~leaves) = 1:nInt;
            renameV(leaves) = nInt+1:n_;
            
            %set the new adj matrix
            obj.adj = adjMat;
            %arrange the visible vector according to the rename used
            obj.v(renameV) = obj.v;
            newObj = obj;
        end
        
        function t2 = getSpecularTree(t1)
            t2 = t1;
            L_ = t1.L;
            newAdj = t2.adj;
            newAdj = L_ + 1 - newAdj;
            newAdj = mod(newAdj,L_+1);
            t2.adj = newAdj;
        end

        function plot(t,vocabulary)
            dNode = 300;
            if(nargin < 2)
                vocabulary = [];
            end
            [~,parents] = max(t.adj);
            
            parents(1)=0;
            
            [x,y] = treelayout(parents);
            hold on;
            
            %draw edges
            for i=1:t.n-t.nf
                setCh = find(t.adj(i,:));
                for ch = setCh
                    %draw edge
                    line([x(i) x(ch)],[y(i) y(ch)],'Color','k');
                end
            end
            
            %draw nodes
            scatter(x,y,dNode,'filled','MarkerFaceColor','w','MarkerEdgeColor','k');

            %draw label
            for i=1:t.n
                if(isempty(vocabulary))
                    s = num2str(t.v(i));
                else
                    s = vocabulary{t.v(i)};
                end
                text(x(i),y(i),s,'HorizontalAlignment','center','VerticalAlignment','middle')
                if(i~=1)
                    pa = t.adj(:,i)~=0;
                    pos = t.adj(pa,i);
                else
                    pos = 0;
                end
                text(x(i)-0.03,y(i)+0.03,num2str(pos),'HorizontalAlignment','center','VerticalAlignment','middle')
            end
            axis off;
        end
        
        function ch_list = get_children_list(t,u)
            ch_list = - ones(t.L,1);
            [~,ch_id,pos_id] =  find(t.adj(u,:));
            ch_list(pos_id) = ch_id;
        end
        
        function pos_u = get_position(t,u)
            [~,~,pos_u] = find(t.adj(:,u));
        end
        
        function ris = is_leaf(t,u)
            ris = (u > t.n - t.nf);
        end
    end
    
end