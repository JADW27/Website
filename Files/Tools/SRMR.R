srmr <- function (data1, data2)
{
    if(nrow(data1)==nrow(data2)&&ncol(data1)==ncol(data2))
    {
        n <- nrow(A1)                                   #number of people
        j <- ncol(A1)                                   #number of items
        k <- j*(j+1)/2                                  #number of inter-item correlations
        
        COR1 <- cor(data1)                              #inter-item correlation of data 1
        COR2 <- cor(data2)                              #inter-item correlation of data 2
        U1 <- matrix(0,nrow=nrow(COR1),ncol=ncol(COR1)) #initialize upper.tri matrix data 1
        U2 <- matrix(0,nrow=nrow(COR2),ncol=ncol(COR2)) #initialize upper.tri matrix data 2
        for(i in 1:nrow(COR1))
            for(j in 1:ncol(COR1))
        if(upper.tri(COR1,diag=TRUE)[i,j]!=0)
        {U1[i,j] <- COR1[i,j]}                          #input values for upper.tri data 1
        for(i in 1:nrow(COR2))
            for(j in 1:ncol(COR2))
                if(upper.tri(COR2,diag=TRUE)[i,j]!=0)
                {U2[i,j] <- COR2[i,j]}                  #input values for upper.tri data 1
        
        U <- U1 - U2                                    #differences between upper.tri matrices
        
        #calculate SRMR value
        
        H <- 0                                          #initialize H
        
        for(i in 1:length(U))
        {
            L <- U[i]^2
            H <- H+L                                    #sums the squared deviation values for each
        }                                               #inter-item correlation
        
        SRMR <- sqrt(H/k)                               #defines final srmr value
        
        return(SRMR)
    }else stop("Please make sure the two data sets are the same size")
}
