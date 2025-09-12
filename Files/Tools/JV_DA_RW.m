function [Results, DA, CD, GD, Dij, ST, RW] = JV_DA_RW(M,n,ci,save)
%M is the correlation matrix.
%n is the sample size.
%ci is the desired confidence interval percentage between 0 and 100. 
%save: Make last input 'y' if you want to save the DA results to an excel
       %spreadsheet that looks like Table 3 in Azen % Budescu 2003 paper. 
%Results is the joint variance estimates in center column, with confidence
       %intervals in the left and right columns.
%DA is the dominance analysis matrix with additional contributions.
%CD is the conditional dominance matrix containing averages for subsets.
%GD is the general dominance matrix, which averages the averages in CD.
%Dij(i,j) is 1 when i completely dominates j, 0 when j completely dominates
       %i, and 0.5 on the diagonal and when dominance is undetermined. 
%ST is the summary table for dominance analysis when save = 'y'.
%RW is the relative weights vector (percents of total R^2).
          
[rn,cn]=size(M); iv = rn-1; %# of independent variables
if rn == cn
    Y = M(2:end,1); X = M(2:end,2:end);
    ng=[]; %number of groups (1st element is iv, 2nd is # of pairs, 3rd is # of threes, etc.)
    C = {}; %C{3} contains matrix that is (iv,3) with each row being a different group of 3
    v = 1:1:iv; 
    for z = 1:iv
       ng = vertcat(ng, nchoosek(iv,z));
       C{z} = nchoosek(v,z);
    end  
    sp = sum(ng); %size of phi
    
    %Find all R and Beta.
    R = []; Beta = {};
    for a = 1:iv 
        Ra = zeros(ng(a),1); Ba = {};
        for z = 1:ng(a)
            j = C{a}(z,:);
            y = zeros(length(j),1);
            for aa = 1:length(j)
                y(aa) = Y(j(aa));
            end
            x = eye(a);
            for i = 2:a
                for k = 1:i-1;
                    x(i,k) = X(j(i),j(k));
                end
            end   
            x = x + transpose(tril(x,-1));
            B = inv(x)*y;
            Ra(z) = sqrt(y'*B);
            Ba{z} = B;
        end
        R = vertcat(R,Ra);  
        Beta{a} = Ba;
    end
    
    %Relative Weights
    [~,D2,Q] = svd(X);
    D = sqrtm(D2);
    L = Q*D*Q'; %Correlations between original variables (x, rows) and orthogonal variables (z, columns)
    Be = inv(L)*Y; %Correlations between the orthogonal variables (z) and the criterion variable
    E = L.^2*Be.^2;
    RW = E/R(sp)^2;
   
    %Dominance Analysis
    DA = zeros(sp,iv);
    for a = 1:iv
        for z = 1:ng(a)
            i = sum(ng(1:a-1))+z; %The zth group of a is in row i of DA.
            for h = 1:iv %The column h corresponds to single variable.
                t = C{a}(z,:);
                if ismember(h,t) == 0
                    Comb = sort(horzcat(h,t));
                    for aa = 2:iv
                        for zz = 1:ng(aa)
                            tt = C{aa}(zz,:);
                            if length(Comb) == length(tt)  
                                if Comb == tt
                                    ii = sum(ng(1:aa-1))+zz;
                                    DA(i,h) = R(ii)^2 - R(i)^2;
                                end
                            end
                        end
                    end    
                end
            end
        end
    end    
    
    CD = zeros(iv); %Conditional dominance (average over each group size)
    CD(1,:) = transpose(R(1:iv).^2);
    for a = 1:iv-1
        DAS = DA(sum(ng(1:a-1))+1:sum(ng(1:a)),:); %DA subset containing only groups of a
        CD(a+1,:) = sum(DAS)./sum(DAS~=0);
    end    
    
    GD = mean(CD); %General dominance (average of the CD averages)
    
    Dab = vertcat(CD(1,:),DA); %Dab adds null row to top of DA.
    Dij = 0.5*ones(iv); %Dij starts with 0.5 in every cell. Dij is for complete dominance.
    for i = 1:iv
        for j = 1:iv
            cv = zeros(sp+1,1); %Comparison vector
            cv(Dab(:,i)==0)=1; %Finds rows where DA for variable i is 0 (to ignore them).
            cv(Dab(:,j)==0)=1; %Finds rows where DA for variable j is 0 (to ignore them).
            comp = Dab(:,i) > Dab(:,j); %Comparison puts 1's in each row where i dominates j.
            cv(comp==1)=1;
            if cv ==1
                Dij(i,j) = 1;
                display(strcat('X',num2str(i),' completely dominates X',num2str(j),'.'));
            end
            cv = zeros(sp+1,1); cv(Dab(:,i)==0)=1; cv(Dab(:,j)==0)=1;
            comp = Dab(:,i) < Dab(:,j);
            cv(comp==1)=1;
             if cv ==1
                Dij(i,j) = 0;
             end
        end    
    end        
    
    Dc = 0.5*ones(iv); %Dc starts with 0.5 in every cell. Dc is for conditional dominance.
    for i = 1:iv
        for j = 1:iv
            comp = CD(:,i) > CD(:,j); %Comparison puts 1's in each row where i dominates j.
            if comp == 1
                Dc(i,j) = 1;
            elseif comp == 0
                Dc(i,j) = 0;
            end
        end    
    end    
    
    for i = 1:iv
        for j = 1:iv
            if Dc(i,j)*Dij(i,j) == 0.5
                display(strcat('X',num2str(i),' conditionally dominates X',num2str(j),'.'));
            end
        end
    end
     
    Dg = 0.5*ones(iv); %Dg starts with 0.5 in every cell. Dg is for general dominance.
    for i = 1:iv
        for j = 1:iv
            comp = GD(i) > GD(j); %Comparison returns 1 where i dominates j.
            if comp == 1
                Dg(i,j) = 1;
            elseif comp == 0
                Dg(i,j) = 0;
            end
        end    
    end
    
    for i = 1:iv
        for j = 1:iv
            if Dc(i,j)*Dg(i,j) == 0.5
                display(strcat('X',num2str(i),' generally dominates X',num2str(j),'.'));
            end
        end
    end
    
    if save =='y'
        [ST] = Table(DA,CD,GD,iv,ng,R,C);
    else
        ST = 0;
    end    
    
    %Joint Variance
    COR = zeros(sp,iv); %COR b/w singles and everything
    COR(1:iv,1:iv) = (X);
    for a = 2:iv
        for z = 1:ng(a)
            i = sum(ng(1:a-1))+z; %The zth group of a is in row i of COR.
            for h=1:iv %The column h corresponds to single variable.
                t = C{a}(z,:);
                if ismember(h,t) == 1
                    COR(i,h) = Y(h)/R(i);
                else
                    for k = 1:length(t)
                        COR(i,h) = COR(i,h) + (Beta{a}{z}(k)*X(h,t(k)));
                    end
                    COR(i,h) = COR(i,h)/R(i);
                end    
            end
        end
    end
    
    CP = eye(sp); %Correlations between everything higher than singles 
    CP(:,1:iv) = tril(COR);
    for aa = 2:iv-1
    for a = 2:iv
        for i = sum(ng(1:a-1))+1:sum(ng(1:a))
            z = i-sum(ng(1:a-1)); %The zth group of a is in row i of CP.
            for c = sum(ng(1:aa-1))+1:sum(ng(1:aa))
                if c < i 
                h = c-sum(ng(1:aa-1)); %The hth group of aa is in column c of CP.
                t = C{a}(z,:); %The group in row i.
                u = C{aa}(h,:); %The group in column c.
                for k = 1:length(t)
                    for j = 1:length(u)
                        CP(i,c) = CP(i,c) + (Beta{a}{z}(k)*Beta{aa}{h}(j)*X(u(j),t(k)));
                    end  
                end
                CP(i,c) = CP(i,c)/(R(i)*R(c));
                else
                end    
            end
        end    
    end
    end
        
    phi = zeros(sp);
    for z = 1:sp
        phi(z,z) = (1-R(z)^2)^2/n;
    end
    
    for y = 2:sp
        for z = 1:y-1
            phi(y,z) = ((.5*(2*CP(y,z)-R(y)*R(z))*(1-R(y)^2-R(z)^2-CP(y,z)^2))+CP(y,z)^3)/n; %Ask Justin about R2 vs. X for 2nd to last thing.
        end
    end
    
    phi = phi + transpose(tril(phi,-1));
    
    AW =zeros(sp); 
    for y = 1:sp-1
        for z = 1:sp-1
            if y+z == sp
                AW(y,z) = -2;
            end
        end
    end
    
    for aa = 1:iv-1
        for c = sum(ng(1:aa-1))+1:sum(ng(1:aa))
              f = find(AW(:,c));
            for a = 1:iv
                for i = sum(ng(1:a-1))+1:sum(ng(1:a))
                    if i == f
                        z = i-sum(ng(1:a-1)); %The zth group of a is in row i of CP. 
                        Ex = C{a}(z,:);
                        se = length(Ex);
                    end
                end
            end
            
            for a = se+1:iv
                for i = sum(ng(1:a-1))+1:sum(ng(1:a))
                    z = i-sum(ng(1:a-1)); %The zth group of a is in row i of CP.
                    t = C{a}(z,:); %The group in row i.
                    if ismember(Ex,t) == 1 
                        if mod((a-se),2) == 0 %if a - se is even 
                            AW(i,c) = -2;
                        else
                            AW(i,c) = 2;
                        end    
                    end  
                end
            end    
        end
    end
    
    if mod(iv,2) == 0 
        AW(sp,sp) = -2;
    else
        AW(sp,sp) = 2;
    end
    
    AW = tril(AW) + transpose(tril(AW,-1));
    
    A = AW.*repmat(R',sp,1);
    Var = A*phi*A';
    Vd = diag(Var);
    cm = norminv([(1-ci/100)/2 (1+ci/100)/2],0,1);
    CI = cm(2)*sqrt(Vd);
        
    JV = zeros(sp,1);
    for z = 1:sp
        JV(z) = (R.^2)'*AW(:,z)/2;
    end
    
    Results = zeros(sp,3);
    Results(:,1) = JV-CI; Results(:,2) = JV; Results(:,3) = JV+CI;
else
    diplay ('Please make sure your correlation matrix has equal numbers of columns and rows.')
    Results = 0;
end    