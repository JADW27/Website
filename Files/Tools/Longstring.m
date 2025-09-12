%First row of data file contains column headers. Examples:
%S1I1 = sub-scale 1, item 1
%S2I5(R) = sub-scale 2, item 5, reversed scoring
%Bogus1 = bogus item 1
%Inst2 = instructed item 2
%SR1 = self-report item 1 (related to data quality)
%RT = response time
%ID = participant identification number

function [LS]=Longstring(DataFile)
%Make sure DataFile and this M-file are in same folder. 
%Otherwise, specify full file path ('C:\...').
%As an example, if your data file is called RawData.txt, call this function
%in the main Matlab window by typing: [LS]=Longstring('RawData.txt');

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

id=strfind(C,'ID'); %Search for ID column header.
rt=strfind(C,'RT'); %Search for response time column header.
ID=cellfun('isempty',id);
RT=cellfun('isempty',rt);

for z=1:NC
    if ID(z)==0 || RT(z)==0 %If column z is bogus, instructed, self-response, or response time
       B(:,z)= zeros(NR,1); %replace column z with zeros.
    end
end
B(:,~any(B,1))=[]; %Remove columns with all 0's from B.
%Now B is the matrix of data excluding ID numbers and response times.

NI=size(B,2); %Number of items

LC=zeros(NR,NI); %Longstring counter initially populated with all 0's
LC(:,1)=1; %First column of longstring counter is all 1's.
for z=2:NI
    for y=1:NR
        d=(B(:,z)==B(:,z-1)); %Check if two consecutive items are equal.
        if d(y)==1
            LC(y,z)=LC(y,z-1)+1; %Add 1 to counter if they are equal.
        else
            LC(y,z)=1; %If they aren't equal, counter goes back to 1.
        end
    end
end

LS=max(LC,[],2); %Longstring index vector. 
figure;
p=plot(LS,'s'); set(p,'MarkerEdgeColor','none','MarkerFaceColor','b');
title('Longstring','FontSize', 18);
ylabel('Longstring Index'); xlabel('Respondent Number');

FileOut=strcat('Longstring',DataFile);

if k>0
    xlswrite(FileOut, LS);
else
    dlmwrite(FileOut, LS);
end
end