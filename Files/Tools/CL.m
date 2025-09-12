%Before beginning, please check the following:
%DataFile1 and DataFile2 contain only responses, no column or row headings (e.g., ID numbers or variable names).
%Items and respondents are in the same order in the 2 files.
%Each item is a column; each respondent is a row.
%No data are missing. Data sets are the same size.
%Both data files have the same file extension (e.g., .txt, .xlsx, .csv).

%Defines the CL_TC function. 
function [CL_TC]=CL(DataFile1, DataFile2)
%For example,if the files are called DataTime1.txt and DataTime2.txt, run this function by typing: [CL_TC]=CL('DataTime1.txt','DataTime2.txt');
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

%This section calculates the inter-item correlation matrices and corresponding eigenvalues and eigenvectors. 
E1=mean(A1); E2=mean(A2); %Calculates the means for each item.
F1=repmat(E1,NR,1); F2=repmat(E2,NR,1); %Replicates the mean values so that matrix dimensions agree.
X1=A1-F1; X2=A2-F2; %Calculates matrices X1 and X2 (centered scores).
G1=std(A1); G2=std(A2); %Calculates the standard deviations for each item.
S1=repmat(G1,NR,1); S2=repmat(G2,NR,1); %Replicates the standard deviation values so that matrix dimensions agree.
Z1=X1./S1; Z2=X2./S2; %Calculates matrices Z1 and Z2 (standardized scores).
COR1=transpose(Z1)*Z1/(NR-1); COR2=transpose(Z2)*Z2/(NR-1);  %Calculates matrices COR1 and COR2 (inter-item correlation matrices).
V1=eig(COR1); V2=eig(COR2);%Calculates V1 and V2 (vectors of eigenvalues).
[W1,D1]=eig(COR1); [W2,D2]=eig(COR2); %Calculates W1 and W2 (matrices with columns containing eigenvectors) as well as D1 and D2 (matrices with the eigenvalues on the diagonal). 

%This section calculates the correlations between component loadings and corresponding weights.
L1=W1*sqrt(D1); L2=W2*sqrt(D2); %Calculates L1 and L2 (component loadings matrices).
R=diag(corr(L1,L2)); %Calculates correlations between component loadings.
w=(V1+V2)/(2*j); %Calculates component weights.

%This section calculates the CL_TC value.
H=0; 
for i=1:j
    N=R(i)^2*w(i); 
    H=H+N; %Sums the (R^2*weight) elements.
end
CL_TC=H; %Defines the final CL_TC value. 

else
    disp('Please make sure the two data sets are the same size.');
    CL_TC=NaN; %This displays an error message if the two data matrices do not have equal dimensions and sets CL_TC to an undefined value.
end

end