%Before beginning, please check the following:
%DataFile1 and DataFile2 contain only responses, no column or row headings (e.g., ID numbers or variable names).
%Items and respondents are in the same order in the 2 files.
%Each item is a column; each respondent is a row.
%No data are missing. Data sets are the same size.
%Both data files have the same file extension (e.g., .txt, .xlsx, .csv).

%Defines the D2ptc function.
function [D2ptc]=D2ptc(DataFile1, DataFile2)
%For example,if the files are called DataTime1.txt and DataTime2.txt, run this function by typing: [D2ptc]=D2ptc('DataTime1.txt','DataTime2.txt');
%Omit semi-colon to display result.

%This section reads the data files.
a=strfind(DataFile1, '.xls'); 
if a>0
    A1=xlsread(DataFile1);
    A2=xlsread(DataFile2);
else
    A1=importdata(DataFile1);
    A2=importdata(DataFile2);
end

%This section checks to see that two data sets are the same size.
if size(A1)==size(A2)

NR=size(A1,1); %NR = Number of respondents.
j=size(A1,2); %j = Number of items.

%This section calculates the D2ptc scores.
X=A1-A2; %Calculates X (matrix of difference scores).
DIF=transpose(X)*X/(NR-1); %DIF is the difference matrix (DIFxx).
DIF_1=inv(DIF); %DIF_1 is the inverted difference matrix (DIFxx^-1).
G=X*DIF_1*transpose(X); %Calculates the D2ptc matrix (individual D2ptc scores appear on the diagonal).
D2ptc=diag(G); %Sets the "D2ptc" variable to equal the diagonal of the D2ptc matrix. 

%This section plots D2ptc scores for visual comparison.
figure;
p=plot(D2ptc,'s'); set(p,'MarkerEdgeColor','none','MarkerFaceColor','b');
title('Personal Temporal Consistency','FontSize', 18);
ylabel('D^2 Value'); xlabel('Respondent Number');

%This section outputs a file containing a vector of respondent D2ptc scores.
FileOut='D2ptc';
if a>0
    xlswrite(FileOut, D2ptc);
else
    dlmwrite(FileOut, D2ptc);
end

else
    disp('Please make sure the two data sets are the same size.');
    D2ptc=NaN; %This displays an error message if the two data matrices do not have equal dimensions and sets D2ptc to an undefined value.
end
end