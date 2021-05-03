library(tidyverse)
library(tm)
library(readr)

listings <- read_csv("C:/Users/324910/Documents/Projects/Airbnb/Data/listings.csv.gz")

## Clean text columns ------------------------------------------------------

listings$description <- str_replace_all(listings$description, regex(c("\\<.*\\>")), " ")

listings$description <- tolower(listings$description)
listings$description <- tm::removePunctuation(listings$description)
listings$description <- tm::removeNumbers(listings$description)
listings$description <- tm::removeWords(listings$description, stopwords("english"))
listings$description <- tm::stripWhitespace(listings$description)

listings$neighborhood_overview <- str_replace_all(listings$neighborhood_overview, regex(c("\\<.*\\>")), " ")

listings$neighborhood_overview <- tolower(listings$neighborhood_overview)
listings$neighborhood_overview <- tm::removePunctuation(listings$neighborhood_overview)
listings$neighborhood_overview <- tm::removeNumbers(listings$neighborhood_overview)
listings$neighborhood_overview <- tm::removeWords(listings$neighborhood_overview, stopwords("english"))
listings$neighborhood_overview <- tm::stripWhitespace(listings$neighborhood_overview)

listings$desc_unique <- quanteda::ntype(listings$description)
listings$neigh_unique <- quanteda::ntype(listings$neighborhood_overview)

# write uncompressed data
listings %>% write_csv("C:/Users/324910/Documents/Projects/Airbnb/Data/listings_cleaned.csv.gz")

# LDA ---------------------------------------------------------------------

#load topic models library
library(topicmodels)

#Set parameters for Gibbs sampling
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

#Number of topics
k <- 3

#Run LDA using Gibbs sampling
ldaOut <- topicmodels::LDA(dtm,k, method = "Gibbs", 
              control=list(nstart=nstart, seed = seed, best=best, 
                           burnin = burnin, iter = iter, thin=thin))

#write out results
#docs to topics
ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics,file=paste("LDAGibbs",k,"DocsToTopics.csv"))

#top 6 terms in each topic
ldaOut.terms <- as.matrix(terms(ldaOut,6))
write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))

#probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma)
write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))


#Find relative importance of top 2 topics
topic1ToTopic2 <- lapply(1:nrow(dtm),function(x)
  sort(topicProbabilities[x,])[k]/sort(topicProbabilities[x,])[k-1])


#Find relative importance of second and third most important topics
topic2ToTopic3 <- lapply(1:nrow(dtm),function(x)
  sort(topicProbabilities[x,])[k-1]/sort(topicProbabilities[x,])[k-2])


#write to file
write.csv(topic1ToTopic2,file=paste("LDAGibbs",k,"Topic1ToTopic2.csv"))
write.csv(topic2ToTopic3,file=paste("LDAGibbs",k,"Topic2ToTopic3.csv"))

