# The script does the following:
# (1) Estimate spatial pattern of mortality intensity using EWS characteristics detected with zero lead time;
# (2) Predict the spatial pattern of mortality intensity using EWS characteristics detected 6 months ahead.
# (3) Generate sub-figures of Fig. 6d, 6e and 6f.


# ------------- set up ------------------
setwd("C:/Users/yanlan/Dropbox/EWSpred")

library(Formula)
#library(abind)
#library(magic)
#library(lattice)
#library(coda)
library(spBayes)
library(gtools)

library(ggplot2)
#library(raster)
#library(pracma)
#library(scales)
#library(ggpubr)

targetyear <- 15
yrrange <- c(targetyear) # Years 2005-2015 was computed, which should be seq(5,15). Here 2015 is used as a demo. 
n.samples <- 100 # A sample size of 2e4 was used, which requires a computing long time. 100 is used for the demo.
LT <- 0 # lead time used for mortality estimation, 0 months
IDtoPlot <- c(1,3,4,5,6,9,12) # species ID
OPTID <- c(2434,336,765,2440,1247,2202,855)  # optimal model for each species from model selection, corresponding to 3rd column in Table S2.
TAG <- "0125d" # spatial estimation and prediction scale, 0.125 degree

lonmin <- -124.5
lonmax <- -116
latmin <- 32.5
latmax <- 42.5
dd <- 1/8
margin = c(-0.1,-0.2,-0.5,-0.3)
clist <- c("snow3","wheat2","firebrick4")


#------- to generate a list of models combinging predictive variables ------------
ELE <-c("ba","dem","dura","area","I(ba*ba)","I(ba*dem)","I(ba*dura)","I(ba*area)",
        "I(dem*dem)","I(dem*dura)","I(dem*area)","I(dura*dura)","I(dura*area)","I(area*area)")
OCCmod <- as.list(ELE)
tmp<-combinations(n=length(ELE),r=2,v=ELE,repeats.allowed=F)
OCCmod <- c(OCCmod,paste(tmp[,1],tmp[,2],sep="+"))
tmp<-combinations(n=length(ELE),r=3,v=ELE,repeats.allowed=F)
OCCmod <- c(OCCmod,paste(tmp[,1],tmp[,2],tmp[,3],sep="+"))
tmp<-combinations(n=length(ELE),r=4,v=ELE,repeats.allowed=F)
OCCmod <- c(OCCmod,paste(tmp[,1],tmp[,2],tmp[,3],tmp[,4],sep="+"))
tmp<-combinations(n=length(ELE),r=5,v=ELE,repeats.allowed=F)
OCCmod <- c(OCCmod,paste(tmp[,1],tmp[,2],tmp[,3],tmp[,4],tmp[,5],sep="+"))


# ----------------------------- parameters for spBayes ------------------------------------
burn.in <- n.samples/5 # burn-in samples
cov.model <- "exponential" # assuming exponential spatial correlation structure
priors <- list("phi.Unif"=3/c(20,0.1), "sigma.sq.IG"=c(2,1),
               "tau.sq.IG"=c(2,1)) # flat prior for beta
starting <- list("phi"=3/5,"sigma.sq"=0.5,"tau.sq"=0.5) # correlation range, 5 degree
tuning <- list("phi"=0.1,"sigma.sq"=0.1,"tau.sq"=0.1) # step size for tunning


# -- Load data containting EWS charateristics and auxiliary information at 0 lead time ---
load(paste("Data/LT",toString(LT),"/grid",TAG,".RData",sep='')) 

# ------------------- Estimation, loop over each species in each year --------------------------------
A <- data.frame(NULL)
for (ss in IDtoPlot){
  print(paste("Species",toString(ss),sep=" "))
  pines <- grid[grid$spid==ss,]
  pines <- pines[pines$mort>0.001,] # threshold of mortality occurrence
  coords <- as.matrix(pines[,c("lon","lat")]
                      +cbind(rnorm(nrow(pines), sd = 0.001),
                             rnorm(nrow(pines), sd = 0.001)))
  pines <- pines[duplicated(coords)==F,]
  coords <- coords[duplicated(coords)==F,]
  
  pines$ba <- (pines$ba-mean(pines$ba))/sd(pines$ba)
  pines$dem <- (pines$dem-mean(pines$dem))/sd(pines$dem)
  pines$dura <- (pines$dura-mean(pines$dura))/sd(pines$dura)
  pines$area <- (pines$area-mean(pines$area))/sd(pines$area)
  
  predictors <- OCCmod[OPTID[which(IDtoPlot==ss)]]
  fm <- formula(paste("log10(mort)~",predictors,sep=''))
  
  m0 <- lm(fm,data=pines)
  pines$resid <- m0$residuals
  pines$m0.fitted <- m0$fitted.values
  spfm <- formula("resid~1")
  
  for (yid in yrrange){ 
    
    tr <- pines[pines$yr==yid,]
    tr.coords <- as.matrix(tr[,c("lon","lat")]
                           +cbind(rnorm(nrow(tr), sd = 0.001),
                                  rnorm(nrow(tr), sd = 0.001)))
    tr <- tr[duplicated(tr.coords)==F,]
    tr.coords <- tr.coords[duplicated(tr.coords)==F,]
    flag <- -1
    if (nrow(tr)>5){
      flag <- 0
      sp.exact <- spLM(spfm,data=tr,coords=tr.coords,n.samples=n.samples,
                       starting=starting,tuning=tuning,priors=priors,cov.model=cov.model,verbose=F)
      bef.sp <- spRecover(sp.exact, start=burn.in)
      beta.samples <- bef.sp$p.beta.recover.samples
      w.samples <- bef.sp$p.w.recover.samples
      beta.hat.mu <- apply(beta.samples,2,median)
      w.hat.mu <- apply(w.samples,1,median)
      tr$sp.fitted <- tr$m0.fitted+w.hat.mu+beta.hat.mu
      A <- rbind(A,tr)
    }
  }
}
#---------------- save an exmaple estimation output using small sample size --------------------
#save(A,file=paste('Output/demo_Est',toString(LT),'_',TAG,'.RData',sep='')) #each row: years, each column: species

#---------------- load actual estimation results using large sample size -----------------------
A <- read.csv(file="Output/Est0_0125d.csv", header=TRUE, sep=",")

#------------ organize data and plot the spatial pattern of estimated intensity ----------------
states <- map_data("state")
ca <- subset(states, region %in% c("california"))
A <- data.frame(A)
A$lat <- latmax-A$ID_R/2-floor(A$supid/4)*dd-dd/2
A$lon <- lonmin+A$ID_C/2+A$supid%%4*dd+dd/2
A$lat <- latmax-A$ID_R/2-floor(A$supid/4)*dd-dd/2
A$lon <- lonmin+A$ID_C/2+A$supid%%4*dd+dd/2
load("Data/LT0/grid0125d.RData")  
groupby <- c("ID_R","ID_C","supid","yr")
tobegrouped <- c("n","mortn")
grid <- aggregate(grid[,names(grid) %in% tobegrouped],as.list(grid[,names(grid) %in% groupby]),sum)
grid$mort <- grid$mortn/grid$n
grid$lat <- latmax-grid$ID_R/2-floor(grid$supid/4)*dd-dd/2
grid$lon <- lonmin+grid$ID_C/2+grid$supid%%4*dd+dd/2

A <- merge(A[,c("lon","lat","yr","sp.fitted.n","mortn")],grid[,c("lon","lat","yr","n")],by=c("lon","lat","yr"),all.y=TRUE)
A$mortn <- log10(A$mortn)
A$mortn[is.na(A$mortn)] <- -3 # no mortality occurrence
A$sp.fitted.n[is.na(A$sp.fitted.n)] <- -3 
lowlim <- -3.1;  uplim <- -0.2
A$mortn[A$mortn>uplim] <- uplim
A$sp.fitted.n[A$sp.fitted.n>uplim] <- uplim

lat <- rep(seq(latmax-dd/2,latmin,-dd),each=8.5*8)
lon <- rep(seq(lonmin+dd/2,lonmax,dd),10*8)
fit <- data.frame(lon,lat)
fit <- merge(A[A$yr==targetyear,c("lon","lat","sp.fitted.n","mortn")],fit,by=c("lon","lat"),all.y=TRUE)
# dfr <- rasterFromXYZ(fit)  #Convert first two columns as lon-lat and third as value                
# plot(dfr)
p.est.2015 <- ggplot(data=fit, aes(x=lon, y=lat)) + geom_raster(aes(fill=sp.fitted.n)) +
  scale_fill_gradient2(low=clist[1], mid=clist[2],high = clist[3],
                       limits=c(lowlim,uplim), midpoint = -1.6,
                       guide="colorbar",na.value="transparent") +
  guides(fill = guide_colorbar(title = "log of mortality intesity",barheight = 2)) +
  geom_polygon(data=ca, aes(x=long,y=lat),fill = NA,color ="black") + 
  coord_equal()+theme_void()+theme(plot.margin = unit(margin,"cm")) +
  theme(plot.title = element_text(size=16, face="bold"))

p.obs.2015 <- ggplot(data=fit, aes(x=lon, y=lat)) + geom_raster(aes(fill=mortn)) +
  scale_fill_gradient2(low=clist[1], mid=clist[2],high = clist[3],
                       limits=c(lowlim,uplim),midpoint = -1.6,
                       guide="colorbar",na.value="transparent") +
  guides(fill = guide_colorbar(title = "",barheight = 2)) +
  geom_polygon(data=ca, aes(x=long,y=lat),fill = NA,color ="black") + 
  coord_equal()+theme_void()+theme(plot.margin = unit(margin,"cm"))+
  theme(plot.title = element_text(size=16, face="bold"))


# -- Load data containting EWS charateristics and auxiliary information with a 6-month lead time ---
LT <- 6 # lead time used for mortality prediction, 6 months
OPTID <- c(2434,834,76,2215,1237,2209,343) # optimal model from model selection
load(paste("Data/LT",toString(LT),"/grid",TAG,".RData",sep='')) 

# ------------------- Prediction, loop over each species in each year ---------------

A <- data.frame(NULL)
for (ss in IDtoPlot){
  print(paste("Species",toString(ss),sep=" "))
  pines <- grid[grid$spid==ss,]
  pines <- pines[pines$mort>0.001,]
  coords <- as.matrix(pines[,c("lon","lat")]
                      +cbind(rnorm(nrow(pines), sd = 0.001),
                             rnorm(nrow(pines), sd = 0.001)))
  pines <- pines[duplicated(coords)==F,]
  coords <- coords[duplicated(coords)==F,]
  
  pines$ba <- (pines$ba-mean(pines$ba))/sd(pines$ba)
  pines$dem <- (pines$dem-mean(pines$dem))/sd(pines$dem)
  pines$dura <- (pines$dura-mean(pines$dura))/sd(pines$dura)
  pines$area <- (pines$area-mean(pines$area))/sd(pines$area)
  
  predictors <- OCCmod[OPTID[which(IDtoPlot==ss)]]
  fm <- formula(paste("log10(mort)~",predictors,sep=''))
  
  m0 <- lm(fm,data=pines)
  pines$resid <- m0$residuals
  pines$m0.fitted <- m0$fitted.values
  spfm <- formula("resid~1")
  
  for (yid in yrrange){
    
    tr <- pines[pines$yr==yid,]
    tr.coords <- as.matrix(tr[,c("lon","lat")]
                           +cbind(rnorm(nrow(tr), sd = 0.001),
                                  rnorm(nrow(tr), sd = 0.001)))
    tr <- tr[duplicated(tr.coords)==F,]
    tr.coords <- tr.coords[duplicated(tr.coords)==F,]
    flag <- -1
    if (nrow(tr)>5){
      sp.exact <- spLM(spfm,data=tr,coords=tr.coords,n.samples=n.samples,
                       starting=starting,tuning=tuning,priors=priors,cov.model=cov.model,verbose=F)
      ts <- pines[pines$yr==yid+1,]
      ts.coords <- as.matrix(ts[,c("lon","lat")]
                             +cbind(rnorm(nrow(ts), sd = 0.001),
                                    rnorm(nrow(ts), sd = 0.001)))
      ts <- ts[duplicated(ts.coords)==F,]
      ts.coords <- ts.coords[duplicated(ts.coords)==F,]
      
      if (nrow(ts)>1){
        flag <- 0
        sp.pred <- spPredict(sp.exact,pred.coords=ts.coords,pred.covars=matrix(rep(1,nrow(ts))))
        sp.pred.mu <- apply(sp.pred$p.y.predictive.samples,1,median)
        ts$sp.fitted <- ts$m0.fitted+sp.pred.mu
        A <- rbind(A,ts)
      }
      
    }
  }
}
#------------- save an prediction exmaple output using small sample size -----------------------
#save(A,file=paste('Output/demo_Prd',toString(LT),'_',TAG,'.RData',sep='')) #each row: years, each column: species


#---------------- load actual estimation results using large sample size -----------------------
A <- read.csv(file="Output/Prd6_0125d.csv", header=TRUE, sep=",")

#------------ organize data and plot the spatial pattern of predicted intensity ----------------
A <- data.frame(A)
A$lat <- latmax-A$ID_R/2-floor(A$supid/4)*dd-dd/2
A$lon <- lonmin+A$ID_C/2+A$supid%%4*dd+dd/2

load("Data/LT0/grid0125d.RData") # grid 
groupby <- c("ID_R","ID_C","supid","yr") # time-varying ecoreg
tobegrouped <- c("n","mortn")
grid <- aggregate(grid[,names(grid) %in% tobegrouped],as.list(grid[,names(grid) %in% groupby]),sum)
grid$mort <- grid$mortn/grid$n
grid$lat <- latmax-grid$ID_R/2-floor(grid$supid/4)*dd-dd/2
grid$lon <- lonmin+grid$ID_C/2+grid$supid%%4*dd+dd/2

A <- merge(A[,c("lon","lat","yr","sp.fitted.n","mortn")],grid[,c("lon","lat","yr","n")],by=c("lon","lat","yr"),all.y=TRUE)
# A$mortn[A$mortn<trd] <- trd

A$mortn <- log10(A$mortn)
A$mortn[is.na(A$mortn)] <- -3
A$sp.fitted.n[is.na(A$sp.fitted.n)] <- -3
# uplim <- -0.1
A$mortn[A$mortn>uplim] <- uplim
A$sp.fitted.n[A$sp.fitted.n>uplim] <- uplim

lat <- rep(seq(latmax-dd/2,latmin,-dd),each=8.5*8)
lon <- rep(seq(lonmin+dd/2,lonmax,dd),10*8)
fit <- data.frame(lon,lat)
fit <- merge(A[A$yr==targetyear,c("lon","lat","sp.fitted.n","mortn")],fit,by=c("lon","lat"),all.y=TRUE)

p.prd.2015 <- ggplot(data=fit, aes(x=lon, y=lat)) + geom_raster(aes(fill=sp.fitted.n)) +
  scale_fill_gradient2(low=clist[1], mid=clist[2],high = clist[3],
                       limits=c(lowlim,uplim), midpoint = -1.6,
                       guide="colorbar",na.value="transparent") +
  guides(fill = guide_colorbar(title = "",barheight = 2)) +
  geom_polygon(data=ca, aes(x=long,y=lat),fill = NA,color ="black") + 
  coord_equal()+theme_void()+theme(plot.margin = unit(margin,"cm")) +
  theme(plot.title = element_text(size=16, face="bold"))


# ------------- Print figures ----------------
p.obs.2015+ ggtitle("Observation, 2015")

p.est.2015+ ggtitle("Estimated, 2015")

p.prd.2015+ ggtitle("Predicted (6 months ahead), 2015")




