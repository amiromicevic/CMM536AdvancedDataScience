# 1. LOAD AND SET UP DATA FRAMES

# Load all libraries required to run the script
library(ROCR)           #Visualizing the Performance of Scoring Classifiers
library(caret)          #Streamline the process for creating predictive models
library(tm)             #Text mining package
library(SnowballC)      #Porter's word stemming algorithm
library(wordcloud)      #Word cloud library
library(RColorBrewer)   #Color schemes for maps and graphics
library(ggplot2)        #Plotting and graphics library
library(e1071)          #Naive Bayes implementation
library(gmodels)        #Cross table evaluation results amongst various model fitting tools
library(nnet)           #Feed-Forward Neural Networks and Multinomial Log-Linear Models

# Ensemble classification svm, slda, boosting, bagging, random forests, glmnet, decision trees, 
# neural networks, maximum entropy
library(RTextTools) 

# For graph and Rgraphviz, if not already installed, use the following 2 commands
# source("http://bioconductor.org/biocLite.R")
# biocLite(c("graph", "RBGL", "Rgraphviz"))

library(graph)
library(Rgraphviz)

# Set working directory, the source file with all user-functions, and load the file
setwd('C:\\Users\\Amir\\Documents\\Coursework\\CMM536AdvancedDataScience')

source("userFunctionsSource.R")

df <- read.csv("leaveRemainTweets_CW.csv")

# Show data frame dimensions (rows and columns) and the names of the columns
nrow(df)
ncol(df)
names(df)

# There is no need to have the user.id column as it serves no purpose
df$user.id = NULL

# Divide the data frame into 2 frames, one where the label is set to Leave, and another for Remain
dfLeave <- subset(df, label=="Leave", select=text:label)
dfRemain <-subset(df, label=="Remain", select=text:label)

# Create constants to be used throughout the script. 
# The change to the value assigned to the constant will propagate to wherever it is being used,
# so only one change is required, rather than multiple chnages
cnstLr <- "Leave and remain"
cnstLea <- "Leave"
cnstRem <- "Remain"

# 2. CUSTOM FUNCTIONS

# A number of custom functions are written to demonstrate code re-usability

# Custom function to generate a text corpus from a text vector
buildTextCorpus <- function(textAsVector)
{
  # Build the text corpus
  textCorpus <- Corpus(VectorSource(textAsVector))
  
  # Please note the use of lazy initialisation where possible. It is a computing standard practice
  # of accessing resources only when needed. It may contribute towards the increase of the overall performance.
  
  # Turn the characters into lower case characters
  textCorpus <- tm_map(textCorpus, content_transformer(tolower))
  
  # Remove punctuation
  textCorpus <- tm_map(textCorpus, removePunctuation,lazy = TRUE)
  
  # Remove numbers
  textCorpus <- tm_map(textCorpus, removeNumbers,lazy = TRUE)
  
  # Remove URLs
  removeURL <- function(x) 
  {
    gsub("http[[:alnum:]]*", "", x)
  }
  
  textCorpus <- tm_map(textCorpus, content_transformer(removeURL))
  
  # Add an extra stop words: 'rt'
  stopWords <- c(stopwords("english"), "rt")
  
  # Remove stopwords from corpus
  textCorpus <- tm_map(textCorpus, removeWords, stopWords)
  
  # Remove white spaces
  textCorpus <- tm_map(textCorpus, stripWhitespace)
  
  # Remove leading and trailing white spaces
  trimLeadingAndTrailing <- function (x) 
  {
    gsub("^\\s+|\\s+$", "", x)
  }
  
  textCorpus <- tm_map(textCorpus, content_transformer(trimLeadingAndTrailing))
  
  # Stem words
  textCorpus <- tm_map(textCorpus, stemDocument)
  
  # Return the clean text corpus to the function caller
  return(textCorpus)
}

# Creates a data frame with terms and frequency from the term document matrix with DESC order
getWordFrequencyFromTdm <- function(tdm)
{
  tFreq <- rowSums(as.matrix(tdm))
  tFreq <- subset(tFreq, tFreq >= 15)
  
  # Store results in a data.frame
  dfFreqTerms <- data.frame(Term = names(tFreq), freq = tFreq)
  
  # Sort the data frame by most frequent words
  dfFreqTerms <- dfFreqTerms[order(-dfFreqTerms$freq),]
  
  return(dfFreqTerms)
}

# Convert data frame of terms and frequency into a table for displaying purpose.
# The table will have a client-driven number of rows.
# This function produces a nicely laid out table with 2 columns,
# as opposed to three unnecessary columns in the frequency terms data frame.
# It also takes a list of terms to be excluded from displaying
getTableFromDataFrame <- function (dfFreqTerms, rowsCount, excludeTerms = NULL)
{
  tfMatrix <- matrix(, nrow = rowsCount, ncol = 2)
  colnames(tfMatrix) <- c("Term", "Frequency")
  rownames(tfMatrix) <- rep("", nrow(tfMatrix)) # Row names are not required
  
  for (i in 1:rowsCount)
  {
    if(is.null(excludeTerms))
    {
      tfMatrix[i, 1] <- as.character(dfFreqTerms$Term[i])
      tfMatrix[i, 2] <- as.character(dfFreqTerms$freq[i])
    }
    else
    {
      
    }
    
  }
  
  return(tfMatrix)
}

# Custom function to display the bar chart of the most frequent words
# Client can determine how many most frequent words are to be displayed 
# by passing in a parameter rowsCount
displayPlot <- function(dfFreqTerms, rowsCount, title)
{
  wordsPlot <- ggplot(dfFreqTerms[1:rowsCount,], aes(x = Term, y = freq))
  wordsPlot <- wordsPlot + geom_bar(stat = "identity")
  wordsPlot <- wordsPlot + xlab( paste(title, "Terms", sep=" ")) + ylab("Frequency") + coord_flip()
  wordsPlot
}

# Generates a word cloud from the given term document matrix
createWordCloud <- function(tdm, title)
{
  m <- as.matrix(tdm)
  v <- sort(rowSums(m), decreasing = TRUE)
  d <- data.frame(word = names(v), freq = v)
  d$word <- gsub("~", " ", d$word)
  
  wordcloud(words = d$word, freq = d$freq, min.freq = 10, title(main=title),
            max.words=2000, random.order=FALSE, rot.per=0.2,
            colors=brewer.pal(8, "Dark2"))
}

# Prints the frequent words
printFrequency <- function(dfTermsFreq, rowsCount, title)
{
  tfMatrix <- getTableFromDataFrame(dfTermsFreq, rowsCount)
  print(title)
  print(tfMatrix, row.names = FALSE, quote=FALSE)

}

# Generates the words associations diagram
visualiseAssociations <- function(tdm, title)
{
  # lowfreq is set to 80 occurences as it offers the best choice of words to provide 
  # a meaningful set of associations 
  fTerms <- findFreqTerms(tdm, lowfreq = 80)
  
  # corThreshold has been reduced to 0.075 to offer the diagram 
  # with the best insight into word associations
  plot(tdm, term = fTerms, corThreshold = 0.075, weighting = T, title(main=title)) 
}

# Generates a text cluster dendrogram of the associated words
visualiseTextClusters <- function(tdm, title)
{
  denseTdm <- removeSparseTerms(tdm, sparse = 0.97) # remove sparse terms
  denseTdmMatrix <- as.matrix(denseTdm)
  
  # compute distance between rows of the matrix
  distMatrix <- dist(scale(denseTdmMatrix))
  
  # cluster using hclust() function
  fit <- hclust(distMatrix, method = "ward.D")
  
  # plot the results
  plot(fit)
  
  # add red rectangles to the plot
  rect.hclust(fit, k = 9)
}

# This level of precision might not be needed, but I will provide seperate answers
# for each data frame: Leave, Remain and Leave/Remain mixed together
# This is done easily through the use of custom functions

# 3. TEXT PRE PROCESSING

# Build the text corpus by calling our custom function for each data frame
textCorpus <- buildTextCorpus(df$text)
textCorpusLeave <- buildTextCorpus(dfLeave$text)
textCorpusRemain <- buildTextCorpus(dfRemain$text)

# Now we shall print out the first 20 rows from the main text corpus to check out
# how well the cleaning process has performed, and if there is anything else
# we might want to do in respect of the text cleaning

for (i in 1:20)
{
  cat(paste("[", i, "] ", sep = ""))
  writeLines(strwrap(as.character(textCorpus[[i]]), width = 100))
}


# 4. CREATE WORD CLOUDS

tdm <- TermDocumentMatrix(textCorpus)
tdmLeave <- TermDocumentMatrix(textCorpusLeave)
tdmRemain <- TermDocumentMatrix(textCorpusRemain)

createWordCloud(tdm, cnstLr)
createWordCloud(tdmLeave, cnstLea)
createWordCloud(tdmRemain, cnstRem)


# 5. GET WORD FREQUENCIES AND ASSOCIATIONS

# 5.1 Find the most frequent word in the collection of tweets
dfTermsFreq = getWordFrequencyFromTdm(tdm)
dfTermsFreqLeave = getWordFrequencyFromTdm(tdmLeave)
dfTermsFreqRemain = getWordFrequencyFromTdm(tdmRemain)

printFrequency(dfTermsFreq, 1, cnstLr)
printFrequency(dfTermsFreqLeave, 1, cnstLea)
printFrequency(dfTermsFreqRemain, 1, cnstRem)

# 5.2 Words association, identify the that words appear together often
visualiseAssociations(tdm, cnstLr)
visualiseAssociations(tdmLeave, cnstLea)
visualiseAssociations(tdmRemain, cnstRem)

# Show the text clusters
visualiseTextClusters(tdm, cnstLr)
visualiseTextClusters(tdmLeave, cnstLea)
visualiseTextClusters(tdmRemain, cnstRem)

# 5.3 The most frequent words that appear in the tweets

# Display the table with top 30 most frequent words
# The number 30 is chosen to provide a wider understanding of the model
printFrequency(dfTermsFreq, 30, cnstLr)
printFrequency(dfTermsFreqLeave, 30, cnstLea)
printFrequency(dfTermsFreqRemain, 30, cnstRem)

# Display the diagram with the top 30 most frequent words
displayPlot(dfTermsFreq, 30, cnstLr)
displayPlot(dfTermsFreqLeave, 30, cnstLea)
displayPlot(dfTermsFreqRemain, 30, cnstRem)

# ADDITIONAL WORK:
# remove the words such as RT @Kitchy65: as they are just twitter user ids
# remove non-alphabetical words such as _ÙàÂ_Ùà¤
# possibly replace . with a dot and an empty space to properly process  future..#Brexit 
# into 2 seperate words, future and Brexit
# possibly re-run the stopwords removal before and after the punctuation removal. i.e. dont still apears 
# in the final textCorpus

# 6. TEXT CLASSIFICATION

# Use one of the 'R' packages to build a classifier that classifies the tweets as leave tweet or remain tweets.
# Notice that in order to do so you are required to complete the following:

# 1. One possible library to use is the RTextTools
# 2. Preprocess and prepare the collection of tweets
# 3. Prepare Document Term Matrix
# 4. You will need to divide your data into training and testing subsets for evaluation of your model
# 5. Report results
# 6. Improve results (i.e. fine-tune your model and re-run the experiment to try to improve the results,
#                     think of using other model/s, or perhaps including more features).

# Naive Bayes approach

# The Bayes classifier only works on categoricial data. 
# This custom function will convert numbers into categorial values
convert_counts <- function(x)
{
  x <- ifelse(x > 0,1,0)
  x <- factor(x,levels = c(0,1),labels = c("No","Yes"))
  return(x)
}

# This function takes in the terms reduction parameter and runs Naive Bayes classifier
# It saves predictions for each lowFreq value passed in
runNaiveBayesClassifier <- function (lowFreqVector, laplaceSmooting = 0)
{
  predictions <- vector("list", length(lowFreqVector))
  
  for(i in 1:length(lowFreqVector))
  {
    # Reduction of the features will be performed by removing terms 
    # that appear less than n times across the documents. Let us experiment to find the best n value.
    freqTerms <- findFreqTerms(dtm, lowFreqVector[i])
    
    # Check the length of the freqTerms/T above. Now we update our training and testing sets
    trainDtmRed <- trainDtm[,freqTerms]
    testDtmRed <- testDtm[,freqTerms]
    
    # Categorise values for Naive Bayes algorithm
    trainDtmRedCat <- apply(trainDtmRed, MARGIN = 2, FUN=convert_counts)
    testDtmRedCat <- apply(testDtmRed, MARGIN = 2, FUN = convert_counts)
    
    # Run Naive Bayes classification algorithm
    set.seed(7)
    bayesModel <- naiveBayes(trainDtmRedCat, trainLabels, laplace = laplaceSmooting)
    
    # Let us evaluate the classifier
    prediction <- predict(bayesModel,testDtmRedCat)
    predictions[[i]] <- prediction
    
  }
  
  return (predictions)
}

# Custom function to create a table from the given dimensions and column names
createTable <- function(nrow, ncol, columnNames)
{
  accMatrix <- matrix(, nrow = nrow, ncol = ncol)
  colnames(accMatrix) <- columnNames
  rownames(accMatrix) <- rep("", nrow(accMatrix)) # Row names are not required
  return(accMatrix)
}

# Display the table indicating the features reduction and the corresponding accuracy
printAccuracyTable <- function(predictions)
{
  accTable = createTable(length(predictions), 2, c("Feature reduction", "Accuracy"))
  for (i in 1:length(predictions))
  {
    prediction <- predictions[[i]]
    
    # Calculate the accuracy metrics from the confusion matrix
    acc <- sum(diag(as.matrix(table(prediction,testLabels))))/length(prediction)
    
    accTable[i, 1] <- as.character(lowFreqVector[i])
    accTable[i, 2] <- as.character(acc)
  }
  
  print(accTable, row.names = FALSE, quote=FALSE)
}

# Display confusion matrix and CrossTable evaluation metrics
printConfusionMatrix <- function(predictions)
{
  for (i in 1:length(predictions))
  {
    prediction <- predictions[[i]]
    
    # Confusion matrix
    print(table(prediction,testLabels))
    
    # Library gmodels provides a function CrossTable for alternative evaluation
    CrossTable(prediction,testLabels,prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted','actual'))
  }
}

# First, let us find out the class/label distribution
table(df$label)

# The DocumentTermMatrix() will now be used to basically pivot the TermMatrixDocument already created
dtm <- DocumentTermMatrix(textCorpus)

# Data will now be split into training and testing data sets with the ratio of 80% v 20%
partitionIndex <- round(nrow(dtm) * 0.8)
trainDtm <- dtm[1:partitionIndex,]
trainLabels <- df[1:partitionIndex,]$label

nextPartitionIndex <- partitionIndex + 1
testDtm <- dtm[nextPartitionIndex:nrow(dtm),]
testLabels <- df [nextPartitionIndex:nrow(dtm),]$label

dim(trainDtm)
dim(testDtm)

# Run Naive Bayes classifier with different feature reduction sizes 
# to find the optimum feature set
lowFreqVector = c(5,7,10,15)
predictions <- runNaiveBayesClassifier(lowFreqVector)
printAccuracyTable(predictions)
printConfusionMatrix(predictions)

# Let us now introduce a Laplacian smoothing parameter to see the effects on the predictions
# We shall use the highest scoring feature reduction size of 10 minimum counts per word across the documents
# and a smoothing parameter set to 0.5
# REF: https://www.quora.com/What-is-Laplacian-smoothing-and-why-do-we-need-it-in-a-Naive-Bayes-classifier
lowFreqVector = c(10)
predictions <- runNaiveBayesClassifier(lowFreqVector, 0.5)
printAccuracyTable(predictions)
printConfusionMatrix(predictions)

#Laplacian smoothing parameter set to 0.5 increases accuracy by roughly 1% to 0.910284463894967

# RTextTools approach
#REF: https://journal.r-project.org/archive/2013/RJ-2013-001/RJ-2013-001.pdf
# Using the prefix rtt for each variable to make it stand out from the other models

# This function creates a data frame containing the algorithms by their performance
# determined by the mean of their balanced F1 score for each label
getAlgorithmsPerformance <- function()
{
  # Create precision recall summary from prediction results
  pr <- create_precisionRecallSummary(rttContainer, rttResults, b_value = 1)
  
  # Extract the mean of the F1 balanced scoress from the precision recall summary
  scores <- c(0,0,0,0,0,0,0)
  startPos <- 3
  for (i in 1:7)
  {
    scores[startPos / 3] <- (pr[(startPos * 2) - 1] + pr[(startPos * 2)])/2
    startPos <- startPos + 3
  }
  
  # Create a final data frame
  scoresDf <- data.frame(Algorithm = c("SVM","SLDA","LOGITBOOST","BAGGING","FORESTS","TREE", "MAXENTROPY"),
                        MeanF1Score = scores)
  
  return(scoresDf)
}

# This function creates a confusion matrix for the given algorithms
getConfusionMatrix <- function(result)
{
  # Convert the 1 and 2 factors into Leave and Remain factors
  resultFactored = factor(result, labels = c("Leave", "Remain"))
  return (confusionMatrix(resultFactored, testLabels))
}


# Create the document term matrix
rttDtm <- create_matrix(df$text, language="english", removeNumbers=TRUE,
                          removePunctuation=TRUE, stripWhitespace=TRUE, toLower=TRUE,
                          removeStopwords=TRUE, stemWords=TRUE, removeSparseTerms=.998)

# Configure the training and testing data
rttContainer <- create_container(rttDtm, as.numeric(factor(df$label)), trainSize=1:partitionIndex,
                              testSize=nextPartitionIndex:nrow(df), virgin=FALSE)

# Train the model with SVM, BOOSTING, MAXENT, RF, TREE, SLDA, BAGGING
set.seed(7)
rttModels <- train_models(rttContainer, algorithms=c("SVM", "BOOSTING", "MAXENT", 
                                                    "RF", "TREE", "SLDA", "BAGGING"))
rttResults <- classify_models(rttContainer, rttModels) 
rttAnalytics <- create_analytics(rttContainer, rttResults, b=1)

# Get the algorithms' performance balanced F score and put them into a data frame
rttFScores <- getAlgorithmsPerformance()

# Sort the data frame in the descending order and print
rttFScores <- rttFScores[with(rttFScores, order(-MeanF1Score)), ]
print(rttFScores,row.names = FALSE, quote=FALSE )

# We can see that the two best performing algorithms are Random forests and Support vector machine
# The following will now display confusion matrix for RF and SVM
for (i in 1:2)
{
  algName <- as.character(rttFScores[i,1])
  parts <-  c('results$', algName, '_LABEL')
  modelLabel <- paste(parts, collapse = '')
  
  # Use eval function to pass in a dynamic parameter converted from the text value in modelLabel
  #https://www.r-bloggers.com/converting-a-string-to-a-variable-name-on-the-fly-and-vice-versa-in-r/
  cat(algName, "", sep = " ")
  print(getConfusionMatrix(eval(parse(text = modelLabel))))
}

# Random forests shows only 30 mis-classified instances out of 457 with the following metrics:
# an overall accuracy of 0.9344, Kappa value of a very high 0.867 
# Leave prediction rate is very high at 0.9639

# Some interesting metrics
# Plot RF probabilities for each classified instance. 
# As the probability value gets closer to 100%, the colour transitions closer to red
rbPal <- colorRampPalette(c('blue','red'))
colours <- rbPal(10)[as.numeric(cut(results$FORESTS_PROB,breaks = 10))]
plot(results$FORESTS_PROB, main="Random Forests Probabilities", ylab="RF probability", col=colours, type="p")
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "#40F0F0")
points(results$FORESTS_PROB, main="Random Forests Probabilities", ylab="RF probability", col=colours, type="p")

# Show the histogram of probabilities
hist(results$FORESTS_PROB, main=names(results$FORESTS_PROB), 
     col = c("#4333FF", "#7B33FF", "#C533FF", "#FF33B1", "#FF339C", "#FF3383", "#FF3361", "#FF334F","#FF2222", "#FF0000"))
lines(density(results$FORESTS_PROB), col="blue", lwd=2) # Add a density estimate with defaults
lines(density(results$FORESTS_PROB, adjust=2), lty="dotted", col="yellow", lwd=2) 

# Show the number of probabilities where P(prediction)>=0.75 and a percentage of it
highProb <- length(results$FORESTS_PROB[results$FORESTS_PROB >= 0.75])
print(highProb)
print(highProb / length(results$FORESTS_PROB))

# Now plot a ROC curve for Random forests algorithm
testLabelsFac <- factor(testLabels, labels = c("1", "2"))
testLabelsFac <- as.numeric(levels(testLabelsFac))[testLabelsFac]
rfLabelsFac <-as.numeric(levels(rttResults$FORESTS_LABEL))[rttResults$FORESTS_LABEL]

rfPrediction <- prediction(rfLabelsFac, testLabelsFac)
plot(performance(rfPrediction,"tpr","fpr"))
rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "#EEFFFF")
plot(performance(rfPrediction,"tpr","fpr"), col="red",  add=TRUE)
plot(performance(rfPrediction,"tpr","fpr"), avg="vertical", spread.estimate="boxplot", add=TRUE)
abline(a=0, b= 1)

# Calculate and show the area under curve
rfAuc <- performance(rfPrediction, measure = "auc")
rfAuc <- rfAuc@y.values[[1]]
rfAuc

# The area under the ROC curve is set to 0.9368109 and is close to 1, which may be interpreted
# as an indication that the algorithm performs very well indeed.
# There is more discussion at https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it

# Let us experiment with the ntree parameter in Random forests algorithm
# to see if we can further improve the performance. We will try with the following 
# values for ntree: 100, 150, 500 and 1000
ntreeExp = c(100,150,500,1000)
for (i in 1:4)
{
  set.seed(7)
  rttModelsExp <- train_models(rttContainer, algorithms=c("RF"), ntree = ntreeExp[i])
  rttResultsExp <- classify_models(rttContainer, rttModelsExp) 
  cat("ntree = ", ntreeExp[i], sep = "")
  cat(" ", "", sep = "")
  print(getConfusionMatrix(rttResultsExp$FORESTS_LABEL))
}

# The best result is achieved by ntree=100 but is no improvement to what we already have

# POTENTIONAL ADDITIONAL WORK: try the neural network
# POTENTIONAL ADDITIONAL WORK: try words correlation a bit more, maybe show a diagram
# POTENTIONAL ADDITIONAL WORK: try to show a diagram to see if the RF algorithm overfits

# CONCLUSION GOES HERE
# The obtained accuracy rate is good given the fairly small amount of data. It is almost senseless to even discuss
# whether the model suffers from the potential overfitting, given the small training and testing data. In the real life
# scenario, we would need a larger text corpus to extract more features into the model, and then test it with a larger
# test data to put it under a proper test. 
# Only then we would be in the position to discuss the real results, bias and variance, and if variance is
# present, then that might indicate a potential overfitting for some test data and the bias/underfitting in other test areas.
# This in a way indicates that we might need the ultimate document terms matrix with all words from the
# specific language in order to get a fully trained model. This sounds computationally expensive and would require
# feature (terms) reduction at a larger scale to make it doable.

# PART 2. DATA STREAMS

# REF: http://data-analytics.net/cep/Schedule_files/Textmining%20%20Clustering,%20Topic%20Modeling,%20and%20Classification.htm
# news <- read.csv("NewsSentiment.csv")

library(XLConnect)
news <- readWorksheetFromFile("RedditNewsExcel.xlsx", sheet=1, startRow = 1, endCol = 2)

# Swap the columns as it is very difficult in the R studio viewer to see the full content of the last columnd
news <- news[c("News", "Date")]


##### SENTIMENT
# REF: http://blog.kaggle.com/2017/10/05/data-science-101-sentiment-analysis-in-r-tutorial/
# REF: http://uc-r.github.io/sentiment_analysis
# REF: http://www.bernhardlearns.com/2017/04/sentiment-analysis-with-r-and-tidytext.html

## Load additional required packages
library(tidyverse)
library(tidytext)
library(glue)
library(stringr)

# Clean up the news data
# Remove the b's and backslashes
news$News <- gsub('b"|b\'|\\\\|\\"', "", news$News)

# Remove punctuation except headline separators
news$News <- gsub("([<>])|[[:punct:]]", "\\1", news$News)

# Remove any dollar signs (they're special characters in R)
news$News <- gsub("\\$", "", news$News)

# Show the size of news data frame after cleanup
dim(news)

# Let us add two new features to the data frame to hold the sentiment values
news["SentimentBing"] <- NA
news["SentimentAfinn"] <- NA

# We shall use two sentiment lexicons and attempt to average out the final semantic value
# which will be set to either positive or negative

#options(warn=-1)
newsRowsCount <- dim(news)[1]
for (i in 1:newsRowsCount)
{
  # Tokenize and remove any dollar signs (they're special characters in R)
  tokens <- data_frame(text = news$News[i]) %>% unnest_tokens(word, text)
  
  # First we use Bing Liu sentiment lexicon
  sentiment <- tokens %>%
    inner_join(get_sentiments("bing")) %>% 
    count(sentiment) %>% 
    spread(sentiment, n, fill = 0) #%>% 

  news[i,3] <- ifelse(is.null(sentiment$negative), 'Pos', 'Neg')

  # Second time around we use Finn Årup Nielsen sentiment lexicon
  sentiment <- tokens %>%
    inner_join(get_sentiments("afinn")) %>% 
    summarise(sentiment = sum(score))
   
  news[i,4] <- sentiment[1]
}

#options(warn=0)

# Let us have a look at the summary of SentimentAfinn column
# This will help us determine some ordinal values for the final sentiment value
summary(news$SentimentAfinn)

# We can see that the mean of -1.408 indicates that the lexicon favour news to be of a
# negative semantics. Only the 3rd quartile begins to show more positive semantics.
# The lowest value is -25, and the highest value is 14

# Let us add now the new feature to the data frame to hold the final sentiment value
# The ordinal values for the final sentiment will be calculated in the following way:

#NegHigh where SentimentAfinn less or equal to -15
#NegMed where SentimentAfinn between -14 and -6 inclusive
#NegLow where SentimentAfinn between -5 and -1 inclusive
#PosLow where SentimentAfinn between 1 and 5 inclusive
#PosMed where SentimentAfinn between 6 and 10 inclusive
#PosHigh where SentimentAfinn greater than 10
# If SentimentAfinn = 0 it will be set to the value of PosLow if SentimentBing is Pos
# If SentimentAfinn = 0 it will be set to the value of NegLow if SentimentBing is Neg

news["FinalSentiment"] <- NA

news$FinalSentiment[which(news$SentimentAfinn <= -15)] <- 'NegHigh'
news$FinalSentiment[which(news$SentimentAfinn > -15 & news$SentimentAfinn <= -6)] <- 'NegMed'
news$FinalSentiment[which(news$SentimentAfinn > -6 & news$SentimentAfinn < 0)] <- 'NegLow'

news$FinalSentiment[which(news$SentimentAfinn > 0 & news$SentimentAfinn <= 5)] <- 'PosLow'
news$FinalSentiment[which(news$SentimentAfinn > 5 & news$SentimentAfinn <= 10)] <- 'PosMed'
news$FinalSentiment[which(news$SentimentAfinn > 10)] <- 'PosHigh'

news$FinalSentiment[which(news$SentimentAfinn == 0 & news$SentimentBing == 'Pos')] <- 'PosLow'
news$FinalSentiment[which(news$SentimentAfinn == 0 & news$SentimentBing == 'Neg')] <- 'NegLow'

# Let us now check that the FinalSentiment is fully populated without any gaps
length(which(news$FinalSentiment != 'NegHigh' & news$FinalSentiment != 'NegMed' & news$FinalSentiment != 'NegLow' 
     & news$FinalSentiment != 'PosLow' & news$FinalSentiment != 'PosMed' & news$FinalSentiment != 'PosHigh'
     & news$FinalSentiment != 'PosLow' & news$FinalSentiment != 'NegLow')) == 0

# Now we shall create the final data set with all features

dowJones <- read.csv("DowJonesIndex.csv")
dim(dowJones)

# The column Close will be dropped as its purpose is served by the Adj.Close column
dowJones$Close <- NULL

# Let us add the 25 news sentiment columns called News1 to News25
for (i in 1:25)
{
  addNewCol = paste("dowJones$News",toString(i), "<- NA", collapse="", sep="")
  eval(parse(text = addNewCol))
}

# Let us now populate the news columns in dowJones from news dataset
dowJonesRowsCount <- dim(dowJones)[1]
for (i in 1:dowJonesRowsCount)
{
  date <- as.character(dowJones$Date[i])
  dowJones[i,7:31] <- news$FinalSentiment[which(news$Date == date)]
}

# Perform a few checks (first, middle and last record) to ensure that dowJones is populated accurately
dowJonesTest <- dowJones[1,7:31]
newsTest <- news$FinalSentiment[which(news$Date == '2016-07-01')]
all(dowJonesTest == newsTest)

dowJonesTest <- dowJones[1000,7:31]
newsTest <- news$FinalSentiment[which(news$Date == '2012-07-12')]
all(dowJonesTest == newsTest)

dowJonesTest <- dowJones[1989,7:31]
newsTest <- news$FinalSentiment[which(news$Date == '2008-08-08')]
all(dowJonesTest == newsTest)

# Let us add the final feature label to be set to 1 if Adj Close value rose or stayed as the same
# And 0 when DJIA Adj Close value decreased.

# First, show the number of rows where value rose or stayed the same, 1078
length(which(dowJones$Adj.Close >= dowJones$Open))

# Now show the number of rows where value decreased, 911
length(which(dowJones$Adj.Close < dowJones$Open))

# Create and populate label, the final class feature
dowJones$Label <- NA

dowJones$Label[which(dowJones$Adj.Close >= dowJones$Open)] <- 1
dowJones$Label[which(dowJones$Adj.Close < dowJones$Open)] <- 0

# Quick check that label is populated correctly
length(which(dowJones$Label == 1)) == length(which(dowJones$Adj.Close >= dowJones$Open))
length(which(dowJones$Label == 0)) == length(which(dowJones$Adj.Close < dowJones$Open))

# Let us remove the date as it is not needed
dowJones$Date <- NULL

# Last check of the date
sapply(dowJones, class)

# Let us convert Label and all News features to factors
dowJones$Label <- as.factor(dowJones$Label)

for (i in 1:25)
{
  columnName = paste("dowJones$News",toString(i), sep="")
  factorColumn = paste(columnName, " <- as.factor(", columnName, ")", collapse="", sep="")
  eval(parse(text = factorColumn))
}

# Check that the conversion to factors worked
sapply(dowJones, class)

# Data is now ready. It is clean, fully populated and we can commence the model building process.
# The objective is to predict Label as a consequence of the feature set that contains daily share totals 
# and the top 25 news for the given date. Label can have two values, 1 or 0.
# If the closing price is greater or equal to the opening price, Label is set to 1, otherwise to 0.

set.seed(7)
validationIndex <- createDataPartition(dowJones$Label, p=0.80, list=FALSE)

# select 20% of the data for validation
validationDataset <- dowJones[-validationIndex,]

# use the remaining 80% of data to training and testing the models
trainingDataset <- dowJones[validationIndex,]

# THIS WILL BE REVISITED FOR SOME APPROPRIATE DIAGRAMS
barplot(summary(dowJones$Label))
plot(dowJones[,7], main=names(dowJones)[7])
plot(density(dowJones[,2]), main=names(dowJones)[2])
# END OF

# Evaluate algorithms

# We will use 10-fold cross validation with 3 repeats
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"

# Let us create a formula with the features that will be used for modelling
# First trial is with news features only
featuresToUse <- as.formula("Label~News1+News2+News3+News4+News5+News6+News7+News8+News9+News10+
                            News11+News12+News13+News14+News15+News16+News17+News18+News19+News20+
                            News21+News22+News23+News24+News25")


# LG Logistic Regression
set.seed(7)
fit.glm <- train(featuresToUse, data=trainingDataset, method="glm", metric=metric, trControl=control, na.action=na.omit)

# GLMNET Regularized Logistic Regression
set.seed(7)
fit.glmnet <- train(featuresToUse, data=trainingDataset, method="glmnet", metric=metric, trControl=control, na.action=na.omit)

# KNN k-Nearest Neighbours
set.seed(7)
fit.knn <- train(featuresToUse, data=trainingDataset, method="knn", metric=metric, trControl=control, na.action=na.omit)

# CART Classification and Regression Trees
set.seed(7)
fit.cart <- train(featuresToUse, data=trainingDataset, method="rpart", metric=metric, trControl=control, na.action=na.omit)

# NB Naive Bayes
set.seed(7)
fit.nb <- train(featuresToUse, data=trainingDataset, method="nb", metric=metric, trControl=control, na.action=na.omit)

# SVM Support Vector Machines with Radial Basis Functions
set.seed(7)
fit.svm <- train(featuresToUse, data=trainingDataset, method="svmRadial", metric=metric, trControl=control, na.action=na.omit)

# Compare algorithms
resultsNews <- resamples(list(LG=fit.glm, GLMNET=fit.glmnet, KNN=fit.knn, CART=fit.cart, NB=fit.nb, SVM=fit.svm))
summary(resultsNews)
dotplot(resultsNews)


# Evaluate boosting and bagging classifiers

# Bagged CART
set.seed(7)
fit.treebag <- train(featuresToUse, data=trainingDataset, method="treebag", metric=metric, trControl=control, na.action=na.omit)

# Random Forest
set.seed(7)
fit.rf <- train(featuresToUse, data=trainingDataset, method="rf", metric=metric, trControl=control, na.action=na.omit)

# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(featuresToUse, data=trainingDataset, method="gbm", metric=metric, trControl=control, verbose=FALSE, na.action=na.omit)

# C5.0
set.seed(7)
fit.c50 <- train(featuresToUse, data=trainingDataset, method="C5.0", metric=metric, trControl=control, na.action=na.omit)

# Compare results
ensembleResultsNews <- resamples(list(BAG=fit.treebag, RF=fit.rf, GBM=fit.gbm, C50=fit.c50))
summary(ensembleResultsNews)
dotplot(ensembleResultsNews)

# Top 3 best performing models are NB = 0.5420, SVM = 0.5420 and RF = 0.5403
# Let us now predict against validation data using these 3 models

# NB=fit.nb
set.seed(7)
predictionsNB <- predict(fit.nb, newdata=validationDataset)
confusionMatrix(predictionsNB, validationDataset$Label, positive = "1")

# SVM=fit.svm
set.seed(7)
predictionsSVM <- predict(fit.svm, newdata=validationDataset)
confusionMatrix(predictionsSVM, validationDataset$Label, positive = "1")

# RF=fit.rf
set.seed(7)
predictionsRF <- predict(fit.rf, newdata=validationDataset)
confusionMatrix(predictionsRF, validationDataset$Label, positive = "1")

# Results of the predictions are very low, as expected, but above 50% rate,
# so the models can outdo random guessing, just.

# Let us now try to tune the SVM to see if we can increase the final accuracy
grid <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
                              C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,1, 1.5, 2,5))
fit.svmTuned <- train(featuresToUse, data=trainingDataset, method="svmRadial", metric=metric, 
                 trControl=control, na.action=na.omit, tuneGrid = grid)

fit.svmTuned$finalModel

# Let us now use the tuned SVM to predict
set.seed(7)
predictionsSVMTuned <- predict(fit.svmTuned, newdata=validationDataset)
confusionMatrix(predictionsSVMTuned, validationDataset$Label, positive = "1")

# Accuracy has actually gone down, so the SVM tuning did not bring on any improvements whatsoever.

# Show important features
varImp(fit.svm)


# TRY RANDOM FOREST TUNING HERE


# Second trial is with the Open, High and Low totals of the stock market and news features
featuresToUse <- as.formula("Label~Open+High+Low+News1+News2+News3+News4+News5+News6+News7+News8+News9+News10+
                            News11+News12+News13+News14+News15+News16+News17+News18+News19+News20+
                            News21+News22+News23+News24+News25")

# STOPPED HERE
# LG Logistic Regression
set.seed(7)
fitPlus.glm <- train(featuresToUse, data=trainingDataset, method="glm", metric=metric, trControl=control, na.action=na.omit)

# GLMNET Regularized Logistic Regression
set.seed(7)
fitPlus.glmnet <- train(featuresToUse, data=trainingDataset, method="glmnet", metric=metric, trControl=control, na.action=na.omit)

# KNN k-Nearest Neighbours
set.seed(7)
fitPlus.knn <- train(featuresToUse, data=trainingDataset, method="knn", metric=metric, trControl=control, na.action=na.omit)

# CART Classification and Regression Trees
set.seed(7)
fitPlus.cart <- train(featuresToUse, data=trainingDataset, method="rpart", metric=metric, trControl=control, na.action=na.omit)

# NB Naive Bayes
set.seed(7)
fitPlus.nb <- train(featuresToUse, data=trainingDataset, method="nb", metric=metric, trControl=control, na.action=na.omit)

# SVM Support Vector Machines with Radial Basis Functions
set.seed(7)
fitPlus.svm <- train(featuresToUse, data=trainingDataset, method="svmRadial", metric=metric, trControl=control, na.action=na.omit)

# Compare algorithms
resultsNewsPlus <- resamples(list(LG=fitPlus.glm, GLMNET=fitPlus.glmnet, KNN=fitPlus.knn, CART=fitPlus.cart, 
                                  NB=fitPlus.nb, SVM=fitPlus.svm))
summary(resultsNewsPlus)
dotplot(resultsNewsPlus)

# Bagged CART
set.seed(7)
fitPlus.treebag <- train(featuresToUse, data=trainingDataset, method="treebag", metric=metric, trControl=control, na.action=na.omit)

# Random Forest
set.seed(7)
fitPlus.rf <- train(featuresToUse, data=trainingDataset, method="rf", metric=metric, trControl=control, na.action=na.omit)

# Stochastic Gradient Boosting
set.seed(7)
fitPlus.gbm <- train(featuresToUse, data=trainingDataset, method="gbm", metric=metric, trControl=control, verbose=FALSE, na.action=na.omit)

# C5.0
set.seed(7)
fitPlus.c50 <- train(featuresToUse, data=trainingDataset, method="C5.0", metric=metric, trControl=control, na.action=na.omit)

ensembleResultsNewsPlus <- resamples(list(BAG=fitPlus.treebag, RF=fitPlus.rf, GBM=fitPlus.gbm, C50=fitPlus.c50))
summary(ensembleResultsNewsPlus)
dotplot(ensembleResultsNewsPlus)



# Let us now predict using the top 2 classifiers. Their mean accuracy is as follows:
# GLMNET = 0.8333, LG = 0.8283 THIS IS FOR Open, High, Low plus all news

# GLMNET=fit.glmnet
set.seed(7)
predictionsGLMNET <- predict(fit.glmnet, newdata=validationDataset)
confusionMatrix(predictionsGLMNET, validationDataset$Label, positive = "1")

# LG=fit.glm
set.seed(7)
predictionsLG <- predict(fit.glm, newdata=validationDataset)
confusionMatrix(predictionsLG, validationDataset$Label, positive = "1")

# Let us now introduce a feed-forward neural network and evaluate the model
# After tuning the model with rang, size and decay parameters, the following model was chosen
set.seed(7)
fit.ann = nnet(featuresToUse, data=trainingDataset,rang = 0.001, size=6, maxit=10000,decay=0.3, na.action=na.omit)
predictionsAnn <- predict(fit.ann, newdata=validationDataset, type="class")
predictionsAnn <- as.factor(predictionsAnn)
confusionMatrix(predictionsAnn, validationDataset$Label, positive = "1")

# Show the number of cells in the neural network
fit.ann$n

# Show the number of weights
length(fit.ann$wts)


# library(doParallel)
# cl <- makeCluster(detectCores(), type='PSOCK')
# registerDoParallel(cl)
# # turn off parallel processing
# registerDoSEQ()


# This experimental work has been leading us to the final exercise that will attempt to predict 
# the movement of the stock market closing value in respect of the opening value on the same day.
# We also need to be mindful that this model has to work for streaming classification. 
# So we shall use dowJones to craft a new dataset dowJonesNext. It will have the following features:
# 
# NegHigh, NegMed, NegLow, PosLow, PosMed, PosHigh, DayMinus3Label, DayMinus3CORatio,
# DayMinus2Label, DayMinus2CORatio,DayMinus1Label, DayMinus1CORatio, Label
# 
# The first 6 features NegHigh to PosHigh will be the sum aggregates of the corresponding feature values 
# for each observation/record. DayMinues3Label will be the Label value from the (today - 3 days) observation, 
# and DayMinues3CORatio will be Adj.Close/Open from the (today - 3 days) observation etc...
# The reason for this is that we want to widen the feature scope to try to capture as much model behaviour
# as possible. The features from the previous 3 days consisting of the label and the ratio between 
# the closing and opening stock market value serve as an addition to the news features that already proved to be 
# insufficient for any serious modelling when being used in isolation.

# Frame generation and population
dowJonesNext <- data.frame(NegHigh=integer(), NegMed=integer(), NegLow=integer(), 
                           PosLow=integer(), PosMed=integer(), PosHigh=integer(),
                           DayMinus1Label = character(), DayMinus1CORatio = numeric(), 
                           DayMinus2Label = character(), DayMinus2CORatio = numeric(), 
                           DayMinus3Label = character(), DayMinus3CORatio = numeric(), 
                           Label = character(), stringsAsFactors = FALSE)

newsAggr <- c('NegHigh', 'NegMed', 'NegLow', 'PosLow', 'PosMed', 'PosHigh')
rowsCount <- nrow(dowJones)
for (i in 1:rowsCount)
{
  for (j in 1:6)
  {
    dowJonesNext[i,j] <- length(which(dowJones[i,6:30] == newsAggr[j]))
  }

  if(i <= rowsCount - 3)
  {
    dowJonesNext$DayMinus1Label[i] <- as.character(levels(dowJones$Label[i+1])[dowJones$Label[i+1]])
    dowJonesNext$DayMinus1CORatio[i] <- dowJones$Adj.Close[i+1] / dowJones$Open[i+1] 
    
    dowJonesNext$DayMinus2Label[i] <- as.character(levels(dowJones$Label[i+2])[dowJones$Label[i+2]])
    dowJonesNext$DayMinus2CORatio[i] <- dowJones$Adj.Close[i+2] / dowJones$Open[i+2] 
    
    dowJonesNext$DayMinus3Label[i] <- as.character(levels(dowJones$Label[i+3])[dowJones$Label[i+3]])
    dowJonesNext$DayMinus3CORatio[i] <- dowJones$Adj.Close[i+3] / dowJones$Open[i+3] 
  }
}

dowJonesNext$Label <- as.character(levels(dowJones$Label)[dowJones$Label])

# Let us remove the last 3 records as they will not have values for the previous 3 days
dowJonesNext <- dowJonesNext[1:(rowsCount-3),]

# Data is there, so let us convert two label columns into factors
dowJonesNext$DayMinus1Label <- as.factor(dowJonesNext$DayMinus1Label)
dowJonesNext$DayMinus2Label <- as.factor(dowJonesNext$DayMinus2Label)
dowJonesNext$DayMinus3Label <- as.factor(dowJonesNext$DayMinus3Label)
dowJonesNext$Label <- as.factor(dowJonesNext$Label)

# Let us do a quick check that aggregations are right 25 news columns x 1986 observations = 49650
sum(rowSums(dowJonesNext[1:1986, 1:6])) == 49650


# CLASSIFIERS TRAINING
# Now that dowJonesNext data set is clean and ready, let us evaluate some classifiers
set.seed(7)
validationIndex <- createDataPartition(dowJonesNext$Label, p=0.80, list=FALSE)

# select 20% of the data for validation
validationDataset <- dowJonesNext[-validationIndex,]

# use the remaining 80% of data to training and testing the models
trainingDataset <- dowJonesNext[validationIndex,]

# LG Logistic Regression
set.seed(7)
fitNext.glm <- train(Label~., data=trainingDataset, method="glm", metric=metric, trControl=control, na.action=na.omit)

# GLMNET Regularized Logistic Regression
set.seed(7)
fitNext.glmnet <- train(Label~., data=trainingDataset, method="glmnet", metric=metric, trControl=control, na.action=na.omit)

# KNN k-Nearest Neighbours
set.seed(7)
fitNext.knn <- train(Label~., data=trainingDataset, method="knn", metric=metric, trControl=control, na.action=na.omit)

# CART Classification and Regression Trees
set.seed(7)
fitNext.cart <- train(Label~., data=trainingDataset, method="rpart", metric=metric, trControl=control, na.action=na.omit)

# NB Naive Bayes
set.seed(7)
fitNext.nb <- train(Label~., data=trainingDataset, method="nb", metric=metric, trControl=control, na.action=na.omit)

# SVM Support Vector Machines with Radial Basis Functions
set.seed(7)
fitNext.svm <- train(Label~., data=trainingDataset, method="svmRadial", metric=metric, trControl=control, na.action=na.omit)

# Bagged CART
set.seed(7)
fitNext.treebag <- train(Label~., data=trainingDataset, method="treebag", metric=metric, trControl=control, na.action=na.omit)

# Random Forest
set.seed(7)
fitNext.rf <- train(Label~., data=trainingDataset, method="rf", metric=metric, trControl=control, na.action=na.omit)

# Stochastic Gradient Boosting
set.seed(7)
fitNext.gbm <- train(Label~., data=trainingDataset, method="gbm", metric=metric, trControl=control, verbose=FALSE, na.action=na.omit)

# C5.0
set.seed(7)
fitNext.c50 <- train(Label~., data=trainingDataset, method="C5.0", metric=metric, trControl=control, na.action=na.omit)

# Compare algorithms
resultsNext <- resamples(list(LG=fitNext.glm, GLMNET=fitNext.glmnet, KNN=fitNext.knn, 
                              CART=fitNext.cart, NB=fitNext.nb, SVM=fitNext.svm,
                              BAG=fitNext.treebag, RF=fitNext.rf, GBM=fitNext.gbm, C50=fitNext.c50))
summary(resultsNext)
dotplot(resultsNext)

# The best accuracy is obtained by C50 = 0.5551, GBM = 0.5509, CART = 0.5509
# Let us do predictions with these three classifiers now, and also try SVM and RF
set.seed(7)
predictionsNextC50 <- predict(fitNext.c50, newdata=validationDataset)
confusionMatrix(predictionsNextC50, validationDataset$Label, positive = "1")

set.seed(7)
predictionsNextGBM <- predict(fitNext.gbm, newdata=validationDataset)
confusionMatrix(predictionsNextGBM, validationDataset$Label, positive = "1")

set.seed(7)
predictionsNextCART <- predict(fitNext.cart, newdata=validationDataset)
confusionMatrix(predictionsNextCART, validationDataset$Label, positive = "1")

set.seed(7)
predictionsNextSVM <- predict(fitNext.svm, newdata=validationDataset)
confusionMatrix(predictionsNextSVM, validationDataset$Label, positive = "1")

set.seed(7)
predictionsNextRF <- predict(fitNext.rf, newdata=validationDataset)
confusionMatrix(predictionsNextRF, validationDataset$Label, positive = "1")

# C50 produces the accuracy of 0.5416 

# Let us now introduce a feed-forward neural network and evaluate the model
# After tuning the model with rang, size and decay parameters, the following model was chosen
set.seed(7)
fitNext.ann = nnet(Label~., data=trainingDataset,rang = 2, size=70, maxit=50000,decay=0.00001, na.action=na.omit)

predictionsNextAnn <- predict(fitNext.ann, newdata=validationDs, type="class")
predictionsNextAnn <- as.factor(predictionsNextAnn)
confusionMatrix(predictionsNextAnn, validationDs$Label, positive = "1")

#It only produces accuracy of 0.529

# Let us try to center and scale the dataset
preprocessParams <- preProcess(dowJonesNext[,1:13], method=c("range"))
transformed <- predict(preprocessParams, dowJonesNext[,1:13])
summary(transformed)

library(neuralnet)
transformedNN <- dummyVars(" ~ .", data = transformed)
testFrame <- data.frame(predict(transformedNN, newdata = transformed))

features <- as.formula("Label.1~NegHigh+NegMed+NegLow+PosLow+PosMed+PosHigh+DayMinus1Label.0+DayMinus1Label.1+DayMinus1CORatio+
                        DayMinus2Label.0+DayMinus2Label.1+DayMinus2CORatio+
                        DayMinus3Label.0+DayMinus3Label.1+DayMinus3CORatio")

nn <- neuralnet(features,data=testFrame,hidden=c(6,6),linear.output=T)

# Neural network did not converge, therefore no weights were created rendering prediction impossible


