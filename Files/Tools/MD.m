%First row of data file contains column headers. Examples:
%S1I1 = sub-scale 1, item 1
%S2I5(R) = sub-scale 2, item 5, reversed scoring
%Bogus1 = bogus item 1
%Inst2 = instructed item 2
%SR1 = self-report item 1 (related to data quality)
%RT = response time
%ID = participant identification number

function [D2]=MD(DataFile)
%Make sure DataFile and this M-file are in same folder. 
%Otherwise, specify full file path ('C:\...').
%As an example, if your data file is called RawData.txt, call this function
%in the main Matlab window by typing: [D2]=MD('RawData.txt');

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
COV=transpose(X)*X/(NR-1); %COV is the inter-item covariance matrix.
COV_1=inv(COV); %COV_1 is the inverted inter-item covariance matrix.
G=X*COV_1*transpose(X); 
D2=diag(G); %These diagonal elements are the Mahalanobis D^2 values.

figure;
p=plot(D2,'s'); set(p,'MarkerEdgeColor','none','MarkerFaceColor','b');
title('Mahalanobis D^2','FontSize', 18);
ylabel('D^2 Value'); xlabel('Respondent Number');

FileOut=strcat('Mahalanobis',DataFile);

if k>0
    xlswrite(FileOut, D2);
else
    dlmwrite(FileOut, D2);
end
end