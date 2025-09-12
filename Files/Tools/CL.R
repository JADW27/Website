cl <- function (data1, data2)
{
    if(nrow(data1)==nrow(data2)&&ncol(data1)==ncol(data2))
    {
        n <- nrow(data1)                                       #number of people
        j <- ncol(data1)                                       #number of items
        
        COR1 <- cor(data1)                                  #inter-item correlation of data 1
        COR2 <- cor(data2)                                  #inter-item correlation of data 2
        
        V1 <- eigen(COR1)$values                            #eigenvalues for data 1
        V2 <- eigen(COR2)$values                            #eigenvalues for data 2
        D1 <- matrix(0,nrow=nrow(COR1),ncol=ncol(COR1))
        diag(D1)<-V1                                        #eigenvalues on the diagonal
        D2 <- matrix(0,nrow=nrow(COR2),ncol=ncol(COR2))
        diag(D2)<-V2                                       #eigenvalues on the diagonal
        W1 <- eigen(COR1)$vectors                          #eigenvectors for data 1
        W2 <- eigen(COR2)$vectors                          #eigenvectors for data 2
    
       #calculates the correlations between component loadings    
       L1 <- W1%*%sqrt(D1)
       L2 <- W2%*%sqrt(D2)
       R <- diag(cor(L1,L2))                                #correlations between loadings
       w <- (V1+V2)/(2*j)
       
       H <- 0
       
       for(i in 1:j)
       {
           N <- R[i]^2*w[i]
           H <- H + N
       }
       CL <- H
       
       return(CL)
    }else stop("Please make sure the two data sets are the same size")
}
