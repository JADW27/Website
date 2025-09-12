%First row of data file contains column headers. Examples:
%S1I1 = sub-scale 1, item 1
%S2I5(R) = sub-scale 2, item 5, reversed scoring
%Bogus1 = bogus item 1
%Inst2 = instructed item 2
%SR1 = self-report item 1 (related to data quality)
%RT = response time
%ID = participant identification number

function [pr]=PR_Options(DataFile, Scales, Options)
%Inputs: Scales=number of different scales
%Options=Number of response options (Possible responses are integers 1
%through Options.)
%If different scales have different numbers of options, make Options a
%vector like [5 7 7 9 5], where 5 is the number of options for scales 1 and
%5. If all scales have the same number of options, a single number will
%work.
%Make sure DataFile and this M-file are in same folder. 
%Otherwise, specify full file path ('C:\...').
%As an example, if your data file is called RawData.txt, there are 6 scales, 
%and possible responses are 1-7, call this function
%in the main Matlab window by typing: [pr]=PR('RawData.txt',6,7);

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

NC=size(B,2); %Number of columns
NR=size(B,1); %Number of rows (respondents)

if length(Options) == 1
    Options = repmat(Options, 1, Scales);
end

O=[]; E=[];
for z=1:Scales
    G=[]; CC=[];
    str = num2str(z);
    pattern = strcat('S',str,'I');  
    %Note that item labels should include S1I, S2I, S3I, etc. If they do not, either change labels or change the pattern to search for.
    i=strfind(C, pattern); %Search for items in scale z.
    empty = cellfun('isempty', i);
    for y=1:NC
        if empty(y)==0 %If column y is part of Scale z,
         G=[G B(:,y)]; %add column y to matrix G and
         CC = [CC C(y)]; %add column header y to matrix CC. 
        end    
    end
    f=Options(z) + 1; %Define variable f as 1 plus the number of response options.
    %To undo reverse scoring of an item, subtract score from f.
    r = strfind(CC, '(R)'); %Search for reverse-worded items.
    %Note that these item labels should include (R). If they do not, either change labels or change the pattern to search for.
    emptyCells = cellfun('isempty', r);
    for zz=1:size(G,2)
        if emptyCells(zz)==0 %If item zz is reverse scored,
        G(:,zz)= f-G(:,zz); %replace column zz with corresponding normal scores.
        end
    end
Odd=G(:,1:2:end); %Create matrix of only odd items.
Even=G(:,2:2:end); %Create matrix of only even items
AvgOdd=mean(Odd,2); %Find average response values for odd items.
AvgEven=mean(Even,2); %Find average response values for even items.
O=[O AvgOdd]; %Store odd averages for scale z in matrix O.
E=[E AvgEven]; %Store even averages for scale z in matrix E.
end

H=[];
for z=1:NR
   I = corr(transpose(O(z,:)),transpose(E(z,:))); %Correlation between odd and even averages.
   H=vertcat(H,I); %After loop ends, matrix H contains r_pr values for each respondent.
end   

pr=(2.*H)./(1+H); %Correction using the Spearman-Brown prophesy formula.

figure;
p=plot(pr,'s'); set(p,'MarkerEdgeColor','none','MarkerFaceColor','b');
title('Personal Reliability','FontSize', 18);
ylabel('Personal Reliability Index'); xlabel('Respondent Number');

FileOut=strcat('PR',DataFile);

if k>0
    xlswrite(FileOut, pr);
else
    dlmwrite(FileOut, pr);
end
end