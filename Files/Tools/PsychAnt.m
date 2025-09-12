%First row of data file contains column headers. Examples:
%S1I1 = sub-scale 1, item 1
%S2I5(R) = sub-scale 2, item 5, reversed scoring
%Bogus1 = bogus item 1
%Inst2 = instructed item 2
%SR1 = self-report item 1 (related to data quality)
%RT = response time
%ID = participant identification number

function [PA,IPA,CA]=PsychAnt(DataFile, Cutoff)
%Inputs: Cutoff is the correlation below which item
%pairs are considered to be psychometric antonyms.
%Make sure DataFile and this M-file are in same folder. 
%Otherwise, specify full file path ('C:\...').
%As an example, if your data file is called RawData.txt, call this function
%in the main Matlab window by typing: [PA,IPA,CA]=PsychAnt('RawData.txt',-0.6);

%Outputs: PA=psychometric antonyms index, IPA=item pairs (antonyms), CA=
%correlations (antonyms). IPA and CA have same ordering, so 1st correlation
%in IPA corresponds to the first item pair listed in CA. These are
%displayed as antonym_item_pairs and inter_item_correlations.

%Load data from file into matrix B.
%The first row of column headers is stored as matrix C.
k=strfind(DataFile, '.xls'); %Excel files are opened differently than text files.
if k>0
   [B,header]=xlsread(DataFile);
   C=header;
else
    D=importdata(DataFile);
    B=D.data;
    C=D.colheaders;
end

id=strfind(C,'ID'); %Search for ID column header.
b=strfind(C, 'Bogus'); %Search for bogus items.
i=strfind(C, 'Inst'); %Search for instructed items.
sr=strfind(C, 'SR'); %Search for self-report items related to data quality.
rt=strfind(C,'RT'); %Search for response time.
%Note that these column labels should include the strings 'ID', 'Bogus', 'Inst', 'SR', and 'RT'. If they do not, either change labels or change the pattern to search for.
ID=cellfun('isempty',id);
Bogus = cellfun('isempty', b);
Inst = cellfun('isempty', i);
SR = cellfun('isempty', sr);
RT=cellfun('isempty',rt);

NC=size(B,2); %Number of columns
NR=size(B,1); %Number of rows (respondents)

for z=1:NC
    if ID(z)==0 || Bogus(z)==0 || Inst(z)==0 || SR(z)==0 || RT(z)==0 %If column z is ID, bogus, instructed, self-response, or response time
       B(:,z)= zeros(NR,1); %replace column z with zeros.
    end
end
B(:,~any(B,1))=[]; %Remove columns with all 0's from B.
%Now B is the matrix of data excluding ID numbers, bogus items, instructed items, self-response items related to data quality, and response time.

E=mean(B); %Calculate response value mean for each item.
F=repmat(E,NR,1); %Replicate mean values so that matrix dimensions agree for subtraction.
X=B-F; %Matrix X contains centered scores.
G=std(B); %Calculate response value standard deviation for each item.
S=repmat(G,NR,1); %Replicate standard deviation values so that matrix dimensions agree.
Z=X./S; %Matrix Z contains standardized scores.

COR=transpose(Z)*Z/(NR-1); %COR is the inter-item correlation matrix.
[r,c]=find(COR<Cutoff); %Find row and column indices of all inter-item correlations below the cutoff.

L=length(r);
H=[];
for z=1:L
    I=COR(r(z),c(z)); %Find the correlations that are below the cutoff, which correspond to the row and column indices already found.
    H=vertcat(H,I); %Store these correlations in matrix H.
end

J=[H r c]; %Combine correlations and indices into one matrix for sorting.
[d1,d2] = sort(J(:,1)); %Sorts correlations from lowest to highest.
K=J(d2,:); %Matrix K contains correlations from lowest to highest in first column and the corresponding item indices in the next two columns.
M=K(1:2:end,:); %Eliminates every other row, since all correlations appear twice in K.

n=size(M,1);
R=[];
for z=1:n
    if ismember(M(z,2),R) || ismember(M(z,3),R) %If an item has already been used in antonym pair with more negative inter-item correlation, do not include it.
    else
    R=vertcat(R, M(z,:));
    end
end

s=find(ID==0);
t=find(Bogus==0);
u=find(Inst==0);
v=find(SR==0);
vv=find(RT==0);
W=[s t u v vv];
Y=sort(W,'descend');
w=length(W);
for z=1:w
    C(Y(z))=[]; %Remove ID, bogus, instructed, self-response, and response time labels from matrix C.
end

IPA=[];
S1=[];
S2=[];
nr=size(R,1); 
for z=1:nr
    s1=R(z,2); s2=R(z,3);
    i1=C(s1); i2=C(s2);
    IPA=vertcat(IPA,[i1 i2]); %item pair labels
    S1=horzcat(S1,B(:,s1)); %Put response values from 1st item in each pair into matrix S1.
    S2=horzcat(S2,B(:,s2)); %Put response values from 2nd item in each pair into matrix S2.
end

if size(R)==0;
    disp('There are no item pairs with correlations below the cutoff value.');
    CA=0; IPA=0; PA=0;
else    
CA=R(:,1); %item pair correlations

antonym_item_pairs=IPA %Output:item pair labels. 
inter_item_correlations=CA %Output: item pair correlations

PA=[];
for z=1:NR
   co = corr(transpose(S1(z,:)),transpose(S2(z,:))); %Correlation between S1 and S2.
   PA=vertcat(PA,co); %After loop ends, matrix PA contains a psychometric antonym index for each respondent.
end   

figure;
p=plot(PA,'s'); set(p,'MarkerEdgeColor','none','MarkerFaceColor','b');
title('Psychometric Antonyms','FontSize', 18);
ylabel('Psychometric Antonym Index'); xlabel('Respondent Number');

FileOut=strcat('PsychAnt',DataFile);

if k>0
    xlswrite(FileOut, PA);
else
    dlmwrite(FileOut, PA);
end
end
end
