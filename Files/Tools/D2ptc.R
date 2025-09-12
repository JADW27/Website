D2ptc <- function (data1, data2)
{
        if(nrow(data1)==nrow(data2)&&ncol(data1)==ncol(data2))
        {
            n <- nrow(data1)                                       #number of people
            j <- ncol(data1)                                       #number of items
            
            X <- data1 - data2
            DIF <- t(as.matrix(X))%*%as.matrix(X)/(n-1)
            DIF_1 <- solve(DIF)
            G <- as.matrix(X)%*%DIF_1%*%t(as.matrix(X))
            D2 <- diag(G)
            
            plot(D2, xlab = "Respondent Number", ylab = "D^2 Value",
                       main = "Personal Temporal Consistency",pch=16,col="blue")
            
    return(D2)
        }else stop("Please make sure the two data sets are the same size")
}
