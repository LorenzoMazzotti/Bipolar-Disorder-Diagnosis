library(caret)
library(randomForest)
library(factoextra)
library(ggplot2)
library(cluster)
library(dplyr)
library(corrplot)
library(rpart.plot)
library(reshape2)
library(qgraph)
library(stats)
library(cluster)
library(umap)
library(Rtsne)
library(doParallel)
library(xgboost)
library(viridis)
library(knitr)
library(rmarkdown)

Mental_Data <- read.csv("C:/Users/lorenzo/Desktop/Dataset-Mental-Disorders.csv")

head(Mental_Data)
str(Mental_Data)
colnames(Mental_Data)

# Defining the function to extract numeric values from strings 
extract_numeric <- function(x) {
  num <- gsub(" From 10", "", x)  # Remove the " From 10" text
  as.integer(num)  # Convert to integer
}

Mental_Data$Sexual.Activity <- sapply(Mental_Data$Sexual.Activity, extract_numeric, simplify = "vector")
Mental_Data$Concentration <- sapply(Mental_Data$Concentration, extract_numeric, simplify = "vector")
Mental_Data$Optimisim <- sapply(Mental_Data$Optimisim, extract_numeric, simplify = "vector")

print(head(Mental_Data$Sexual.Activity))
print(head(Mental_Data$Concentration))
print(head(Mental_Data$Optimisim))

# Define binary columns with correct column names
binary_columns <- c('Mood.Swing', 'Suicidal.thoughts', 'Anorxia', 'Authority.Respect', 
                    'Try.Explanation', 'Aggressive.Response', 'Ignore...Move.On', 
                    'Nervous.Break.down', 'Admit.Mistakes', 'Overthinking')

# Apply the binary encoding 
for(column in binary_columns) {
  if(column %in% names(Mental_Data)) {
    Mental_Data[[column]] <- as.integer(ifelse(Mental_Data[[column]] == "YES", 1, 0))
    print(head(Mental_Data[[column]]))  # Checking after encoding
  } else {
    print(paste("Column not found:", column))  # Indicate any columns that aren't found
  }
}

# Correct column names and map ordinal variables
Mental_Data$Sadness <- as.integer(factor(Mental_Data$Sadness, levels = c('Seldom', 'Sometimes', 'Usually', 'Most-Often'), labels = c(0, 1, 2, 3)))
Mental_Data$Euphoric <- as.integer(factor(Mental_Data$Euphoric, levels = c('Seldom', 'Sometimes', 'Usually', 'Most-Often'), labels = c(0, 1, 2, 3)))
Mental_Data$Exhausted <- as.integer(factor(Mental_Data$Exhausted, levels = c('Seldom', 'Sometimes', 'Usually', 'Most-Often'), labels = c(0, 1, 2, 3)))
Mental_Data$Sleep.dissorder <- as.integer(factor(Mental_Data$Sleep.dissorder, levels = c('Seldom', 'Sometimes', 'Usually', 'Most-Often'), labels = c(0, 1, 2, 3)))

# Modify the Expert.Diagnose column to a binary format
Mental_Data$Bipolar_Diagnose <- ifelse(Mental_Data$Expert.Diagnose %in% c("Bipolar Type-1", "Bipolar Type-2"), 1, 0)

# Drop the Patient.Number column
Mental_Data$Patient.Number <- NULL

# Check the balance of the new binary target variable
table(Mental_Data$Bipolar_Diagnose)
# Normalize the specific columns
normalize_columns <- c('Sexual.Activity', 'Concentration', 'Optimisim', 'Sadness', 'Euphoric', 'Exhausted', 'Sleep.dissorder')
Mental_Data[normalize_columns] <- as.data.frame(lapply(Mental_Data[normalize_columns], function(x) (x - min(x)) / (max(x) - min(x))))

#Drop the Patient.Number column
Mental_Data$Patient.Number <- NULL

#Create a copy of the dataset for clustering where we keep expert.diagnose and mood swing
Mental_Data_Cluster <- Mental_Data

#drop the original Expert.Diagnose column since we are not going to use it in supervised
Mental_Data$Expert.Diagnose <- NULL

#Drop the Mood.Swing column for supervised dataset since it's a perfect predictor of bipolar
Mental_Data$Mood.Swing <- NULL

######DESCRIPTIVE

#correlation heatmap

correlation_matrix <- cor(Mental_Data, use = "complete.obs")

corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", addCoef.col = "black", 
         tl.col = "black", tl.srt = 45, 
         # Adjust colorRamp as per your aesthetic preference
         col = colorRampPalette(c("#6D9EC1", "white", "#E46726"))(200))

melted_corr_matrix <- melt(correlation_matrix)

ggplot(data = melted_corr_matrix, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#6D9EC1", high = "#E46726", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(x = '', y = '', title = 'Correlation Matrix')

# Create the correlation network graph 

qgraph(correlation_matrix, 
       layout = "spring", 
       vsize = 6, 
       label.cex = 1.2, 
       label.scale = FALSE, 
       labels = colnames(correlation_matrix), 
       label.norm = "O", 
       title = "Correlation Network Graph",
       label.color = "black") 

#distribution of expert diagnoses

# Summarize the count of each expert diagnosis
expert_diagnosis_counts <- Mental_Data_Cluster %>%
  group_by(Expert.Diagnose) %>%
  summarise(Count = n(), .groups = 'drop')

# Create a pie chart for expert diagnoses
expert_pie_chart <- ggplot(expert_diagnosis_counts, aes(x = "", y = Count, fill = Expert.Diagnose)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  theme_void() +
  theme(legend.position = "bottom") +
  labs(fill = "Diagnose Type", title = "Distribution of Expert Diagnoses")

# Print the pie chart
print(expert_pie_chart)

################SUPERVISED 

# Split the dataset into training and testing sets
set.seed(111)  
indexes <- createDataPartition(Mental_Data$Bipolar_Diagnose, p = 0.80, list = FALSE)
train_set <- Mental_Data[indexes,]
test_set <- Mental_Data[-indexes,]

################### Logistic Regression Model
model <- glm(Bipolar_Diagnose ~ ., data = train_set, family = binomial())

summary(model)

# Predict on the test set
predictions <- predict(model, test_set, type = "response")

predicted_classes <- ifelse(predictions > 0.5, 1, 0)

confusionMatrixData <- confusionMatrix(as.factor(predicted_classes), as.factor(test_set$Bipolar_Diagnose))

# Print the confusion matrix and extract the statistics

print(confusionMatrixData$table)
print(confusionMatrixData$overall['Accuracy'])
print(confusionMatrixData$byClass['Balanced Accuracy'])
print(confusionMatrixData$byClass['Precision'])
print(confusionMatrixData$byClass['Recall'])
print(confusionMatrixData$byClass['F1'])

#############KNN MODEL

train_set$Bipolar_Diagnose <- as.factor(train_set$Bipolar_Diagnose)
test_set$Bipolar_Diagnose <- as.factor(test_set$Bipolar_Diagnose)

# Set up the train control using cross-validation
train_control <- trainControl(method = "cv", number = 30, savePredictions = "final")

set.seed(111)

# Train the KNN model with parameter tuning
knn_model <- train(
  Bipolar_Diagnose ~ ., 
  data = train_set, 
  method = "knn", 
  trControl = train_control,
  tuneLength = 20  
)

# Print the trained KNN model details to understand the best k
print(knn_model)

# Predict on the test set 
knn_predictions <- predict(knn_model, newdata = test_set)

# Evaluate the model's performance using the test 
knn_results <- confusionMatrix(knn_predictions, test_set$Bipolar_Diagnose)

print(knn_results$table)
cat("Balanced Accuracy:", knn_results$byClass['Balanced Accuracy'], "\n")
cat("Precision:", knn_results$byClass['Precision'], "\n")
cat("Recall:", knn_results$byClass['Recall'], "\n")
cat("F1 Score:", knn_results$byClass['F1'], "\n")

###################decision tree

tree_model <- rpart(Bipolar_Diagnose ~ ., data = train_set, method = "class")

summary(tree_model)

# Plot the tree
rpart.plot(tree_model)

# Make predictions on the test set
predictions <- predict(tree_model, newdata = test_set, type = "class")

# Evaluate the model 
confusionMatrixData <- confusionMatrix(as.factor(predictions), as.factor(test_set$Bipolar_Diagnose))

print(confusionMatrixData$table)
print(paste("Accuracy:", confusionMatrixData$overall['Accuracy']))
print(paste("Balanced Accuracy:", confusionMatrixData$byClass['Balanced Accuracy']))
print(paste("Precision:", confusionMatrixData$byClass['Precision']))
print(paste("Recall:", confusionMatrixData$byClass['Recall']))
print(paste("F1 Score:", confusionMatrixData$byClass['F1']))

########## RANDOM FOREST WITH PARAMETER TUNING

train_set$Bipolar_Diagnose <- as.factor(train_set$Bipolar_Diagnose)
test_set$Bipolar_Diagnose <- as.factor(test_set$Bipolar_Diagnose)

set.seed(13)

# parameters to tune
mtry_values <- seq(2, sqrt(ncol(train_set) - 1), by = 1)
nodesize_values <- c(1, 5, 10, 15, 20)
minsplit_values <- c(2, 4, 6, 10)
maxdepth_values <- c(5, 10, 15, 20)

# Initialize tracking variables for the best model
best_model <- NULL
best_accuracy <- 0
best_mtry <- NULL
best_nodesize <- NULL
best_minsplit <- NULL
best_maxdepth <- NULL

# Loop through all combinations of mtry, nodesize, minsplit, and maxdepth
for (mtry in mtry_values) {
  for (nodesize in nodesize_values) {
    for (minsplit in minsplit_values) {
      for (maxdepth in maxdepth_values) {
        # Train the model with current settings
        model <- randomForest(Bipolar_Diagnose ~ ., data = train_set, mtry = mtry, 
                              nodesize = nodesize, ntree = 5000, minsplit = minsplit, 
                              maxdepth = maxdepth)
        
        # Make predictions on the test set
        predictions <- predict(model, newdata = test_set)
        
        # Evaluate the model
        cm <- confusionMatrix(as.factor(predictions), as.factor(test_set$Bipolar_Diagnose))
        accuracy <- cm$overall['Accuracy']
        
        # Update best model if the current model is better
        if (accuracy > best_accuracy) {
          best_model <- model
          best_accuracy <- accuracy
          best_mtry <- mtry
          best_nodesize <- nodesize
          best_minsplit <- minsplit
          best_maxdepth <- maxdepth
        }
      }
    }
  }
}

# Print the best model parameters
print(paste("Best mtry:", best_mtry, "Best nodesize:", best_nodesize, "Best minsplit:", best_minsplit,
            "Best maxdepth:", best_maxdepth, "with Accuracy:", best_accuracy))

# Use the best model to predict and evaluate using confusion matrix
best_predictions <- predict(best_model, newdata = test_set)
best_confusionMatrixData <- confusionMatrix(as.factor(best_predictions), as.factor(test_set$Bipolar_Diagnose))

print(best_confusionMatrixData$table)
cat("Accuracy:", best_confusionMatrixData$overall['Accuracy'], "\n")
cat("Balanced Accuracy:", best_confusionMatrixData$byClass['Balanced Accuracy'], "\n")
cat("Precision:", best_confusionMatrixData$byClass['Precision'], "\n")
cat("Recall:", best_confusionMatrixData$byClass['Recall'], "\n")
cat("F1 Score:", best_confusionMatrixData$byClass['F1'], "\n")

# Print variable importance from the best model
importance_values <- importance(best_model)
print(importance_values)

# Plot the variable importance
varImpPlot(best_model)

##########XGBOOSTING + TUNING 

# Set up parallel processing
num_cores <- detectCores() - 1  
cl <- makeCluster(num_cores)
registerDoParallel(cl)

train_set$Bipolar_Diagnose <- as.factor(as.numeric(as.factor(train_set$Bipolar_Diagnose)) - 1)
test_set$Bipolar_Diagnose <- as.factor(as.numeric(as.factor(test_set$Bipolar_Diagnose)) - 1)

train_matrix <- as.matrix(train_set[, -which(names(train_set) == "Bipolar_Diagnose")])
train_label <- train_set$Bipolar_Diagnose

test_matrix <- as.matrix(test_set[, -which(names(test_set) == "Bipolar_Diagnose")])
test_label <- test_set$Bipolar_Diagnose

# Set up train control for cross-validation
train_control <- trainControl(
  method = "cv", 
  number = 5, 
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Define a tuning grid
tune_grid <- expand.grid(
  nrounds = c(50, 100, 200),  # Number of boosting rounds
  max_depth = c(3, 6, 9, 12),  # Maximum depth of the trees
  eta = c(0.01, 0.05, 0.1, 0.3),  # Learning rate
  gamma = c(0, 0.1, 0.5, 1),  # Minimum loss reduction
  colsample_bytree = c(0.5, 0.7, 0.9, 1.0),  # Subsample ratio of columns
  min_child_weight = c(1, 5, 10),  # Minimum sum of instance weight
  subsample = c(0.5, 0.7, 0.9, 1.0)  # Subsample ratio of training instances
)

# Train the model using caret with parallel processing
set.seed(111)
xgb_tuned_model <- train(
  x = train_matrix, 
  y = train_label, 
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid
)

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# Print the best model parameters
print(xgb_tuned_model$bestTune)

# Make predictions with the best tuned model
best_predictions <- predict(xgb_tuned_model, newdata = test_matrix)

best_predictions <- factor(best_predictions, levels = levels(test_label))
test_label <- factor(test_label, levels = levels(best_predictions))

# Generate the confusion matrix and calculate metrics
confusionMatrixData <- confusionMatrix(best_predictions, test_label)

# Print the confusion matrix and the statistics
print(confusionMatrixData$table)
print(paste("Accuracy:", confusionMatrixData$overall['Accuracy']))
print(paste("Balanced Accuracy:", confusionMatrixData$byClass['Balanced Accuracy']))
print(paste("Precision:", confusionMatrixData$byClass['Pos Pred Value']))
print(paste("Recall:", confusionMatrixData$byClass['Sensitivity']))
print(paste("F1 Score:", confusionMatrixData$byClass['F1']))

###################PCA 

colnames(Mental_Data_Cluster)

Mental_Data_Cluster_pca <- Mental_Data_Cluster[,1:17]  

# Standardizing the data
Mental_Data_Cluster_scaled <- scale(Mental_Data_Cluster_pca)

# Performing PCA
pca_results <- prcomp(Mental_Data_Cluster_scaled, center = TRUE, scale. = TRUE)

summary(pca_results)

# Plotting the explained variance
fviz_eig(pca_results, addlabels = TRUE, ylim = c(0, 100))

############### Elbow Method to Determine the Number of Clusters

scores <- pca_results$x[, 1:8]  #Extracting the first eight principal components

# Elbow method to determine the optimal number of clusters
set.seed(123)
wss <- sapply(1:10, function(k) {kmeans(scores, k, nstart = 10)$tot.withinss})

# Plotting the Elbow Method
plot(1:10, wss, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of Clusters", 
     ylab = "Total Within-Clusters Sum of Squares",
     main = "Elbow Method for Optimal Number of Clusters")

############### PCA KMEANS Clustering

optimal_clusters <- 4 

# Perform KMeans clustering
set.seed(123)
kmeans_result <- kmeans(scores, centers = optimal_clusters, nstart = 10)

expert_diagnoses <- Mental_Data_Cluster$Expert.Diagnose 

scores_df <- data.frame(scores, Cluster = factor(kmeans_result$cluster), Expert_Diagnose = expert_diagnoses)

# Plotting the clusters with expert diagnosis labels
ggplot(scores_df, aes(x = PC1, y = PC2, color = Cluster, shape = Expert_Diagnose)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("red", "green", "blue", "purple")) +  
  scale_shape_manual(values = c(1, 2, 3, 4, 5, 6, 15, 17, 18, 19)) +
  theme_minimal() +
  labs(title = "PCA Projection with K-MEANS Clustering",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Cluster",
       shape = "Expert Diagnose")

# Calculate silhouette score
silhouette_score <- silhouette(kmeans_result$cluster, dist(scores))
fviz_silhouette(silhouette_score)
cat("Average Silhouette Width: ", mean(silhouette_score[, 3]), "\n")

##################### HIERARCHICAL PCA 

scores <- pca_results$x[, 1:8]  
expert_diagnoses <- Mental_Data_Cluster$Expert.Diagnose  

# Performing hierarchical clustering using the "complete" method on the PCA results
hc <- hclust(dist(scores), method = "complete")

# Plot the dendrogram 
plot(hc, main = "Hierarchical Clustering Dendrogram", sub = "", xlab = "")

# Cutting the dendrogram 
clusters_hc <- cutree(hc, k = 4)

pca_df <- data.frame(scores, HC_Cluster = as.factor(clusters_hc), Expert_Diagnose = expert_diagnoses)

# Visualize the PCA results with hierarchical clusters
ggplot(pca_df, aes(x = PC1, y = PC2, color = HC_Cluster, shape = Expert_Diagnose)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("red", "green", "blue", "purple")) +
  #scale_color_manual(values = rainbow(length(unique(pca_df$HC_Cluster)))) +
  scale_shape_manual(values = c(1, 2, 3, 4, 5, 6, 15, 17, 18, 19)) +
  theme_minimal() +
  labs(title = "PCA Projection with Hierarchical Clustering",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Cluster",
       shape = "Expert Diagnose")

# Calculate the Silhouette 
sil_scores_hc <- silhouette(clusters_hc, dist(scores))
plot(sil_scores_hc, main = "Silhouette Plot for Hierarchical Clustering")
cat("Average Silhouette Width: ", mean(sil_scores_hc[, 3]), "\n")

#################UMAP
########KMEANS UMAP

Mental_Data_Cluster_features <- Mental_Data_Cluster[, 1:17]

umap_config <- umap.defaults

# Adjustable UMAP parameters 
umap_config$n_neighbors <- 15  
umap_config$min_dist <- 0.01  
umap_config$n_components <- 2

# Apply UMAP
umap_results <- umap(Mental_Data_Cluster_features, config = umap_config)

# Convert UMAP results to a dataframe for plotting
umap_df <- as.data.frame(umap_results$layout)
colnames(umap_df) <- c("UMAP1", "UMAP2")

#Elbow Method
wss <- sapply(1:10, function(k){
  kmeans(Mental_Data_Cluster_features, centers = k, nstart = 10)$tot.withinss
})

plot(1:10, wss, type="b", pch = 19, frame = FALSE, 
     xlab="Number of Clusters (K)", 
     ylab="Total Within-Cluster Sum of Squares (WSS)", 
     main="Elbow Method for Choosing Optimal K")

k <- 4

#KMEANS

set.seed(123)  
kmeans_result <- kmeans(Mental_Data_Cluster_features, centers = k, nstart = 10)

# Adding the cluster assignments to the UMAP dataframe 
umap_df$Cluster <- as.factor(kmeans_result$cluster)

# Adding the Expert_Diagnose column to the UMAP dataframe for visualization
umap_df$Expert_Diagnose <- Mental_Data_Cluster$Expert.Diagnose

ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Cluster, shape = Expert_Diagnose)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("red", "green", "blue", "purple")) +  # Customize colors
  scale_shape_manual(values = c(1, 2, 3, 4, 5, 6, 15, 17, 18, 19)) +
  theme_minimal() +
  labs(title = "UMAP Projection with K-Means Clustering",
       x = "UMAP Dimension 1",
       y = "UMAP Dimension 2",
       color = "Cluster",
       shape = "Expert Diagnose")

# Step 7: Silhouette analysis to evaluate clustering quality
sil_scores <- silhouette(kmeans_result$cluster, dist(Mental_Data_Cluster_features))
plot(sil_scores)
cat("Average Silhouette Width: ", mean(sil_scores[, 3]), "\n")

#Compute average scores for each cluster
Mental_Data_Cluster$Cluster <- kmeans_result$cluster
relevant_columns <- names(Mental_Data_Cluster)[1:17]  # Exclude non-numeric columns
average_scores <- aggregate(. ~ Cluster, data = Mental_Data_Cluster[, c(relevant_columns, "Cluster")], mean)

print(average_scores)

###########################HEIRARCHICAL UMAP

# Perform hierarchical clustering using the "complete" method on the UMAP results
hc <- hclust(dist(umap_df[, 1:2]), method = "complete")

# Plot the dendrogram to visualize the hierarchy
plot(hc, main = "Hierarchical Clustering Dendrogram", sub = "", xlab = "")

# Cutting the dendrogram
clusters_hc <- cutree(hc, k = 3)

umap_df$HC_Cluster <- as.factor(clusters_hc)

ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = HC_Cluster, shape = Expert_Diagnose)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("red", "green", "blue", "purple")) +
  scale_shape_manual(values = c(1, 2, 3, 4, 5, 6, 15, 17, 18, 19)) +
  theme_minimal() +
  labs(title = "UMAP Projection with Hierarchical Clustering",
       x = "UMAP Dimension 1",
       y = "UMAP Dimension 2",
       color = "Cluster",
       shape = "Expert Diagnose")

# Calculate the Silhouette score for the hierarchical clusters
sil_scores_hc <- silhouette(clusters_hc, dist(umap_df[, 1:2]))
plot(sil_scores_hc, main = "Silhouette Plot for Hierarchical Clustering")

Mental_Data_Cluster$HC_Cluster <- clusters_hc

relevant_columns <- names(Mental_Data_Cluster)[1:17]

# Calculate average scores for each variable within the hierarchical clusters
average_scores_hc <- aggregate(. ~ HC_Cluster, data = Mental_Data_Cluster[, c(relevant_columns, "HC_Cluster")], mean)

print(average_scores_hc)

#########TSNE kmeans

Mental_Data_Cluster_features <- Mental_Data_Cluster[, 1:17]

# Running t-SNE
set.seed(123)  
tsne_results <- Rtsne(Mental_Data_Cluster_features, dims = 2, perplexity = 30, verbose = TRUE, max_iter = 3000)

# Convert t-SNE results to a dataframe for plotting
tsne_df <- as.data.frame(tsne_results$Y)
colnames(tsne_df) <- c("TSNE1", "TSNE2")

# Perform KMeans 
set.seed(123)
kmeans_tsne_result <- kmeans(tsne_df, centers = optimal_clusters, nstart = 20)

tsne_df$Cluster <- factor(kmeans_tsne_result$cluster)
tsne_df$Expert_Diagnose <- factor(Mental_Data_Cluster$Expert.Diagnose)

ggplot(tsne_df, aes(x = TSNE1, y = TSNE2, color = Cluster, shape = Expert_Diagnose)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("red", "green", "blue", "purple")) +  
  scale_shape_manual(values = c(1, 2, 3, 4, 5, 6, 15, 17, 18, 19)) +
  theme_minimal() +
  labs(title = "t-SNE Clustering with K-MEANS",
       x = "t-SNE Dimension 1",
       y = "t-SNE Dimension 2",
       color = "Cluster",
       shape = "Expert Diagnose")

# Calculate Silhouette Score for t-SNE + KMeans

tsne_silhouette <- silhouette(kmeans_tsne_result$cluster, dist(tsne_df[, 1:2]))
plot(tsne_silhouette)
cat("Average Silhouette Width (t-SNE + KMeans): ", mean(tsne_silhouette[, 3]), "\n")

Mental_Data_Cluster_features$Cluster <- kmeans_tsne_result$cluster 
# Calculate the average of each feature in each cluster
cluster_means <- aggregate(. ~ Cluster, data = Mental_Data_Cluster_features, FUN = mean)
print(cluster_means)

##################HIERARCHICAL TSNE

dist_tsne <- dist(tsne_df[, 1:2])

# Perform hierarchical clustering using centroid linkage
hc_tsne <- hclust(dist_tsne, method = "centroid")

# Plotting the dendrogram 
plot(hc_tsne, main = "Hierarchical Clustering Dendrogram", sub = "", xlab = "")

# Cutting the dendrogram 
clusters_hc_tsne <- cutree(hc_tsne, k = 4)

tsne_df$HC_Cluster <- factor(clusters_hc_tsne)

ggplot(tsne_df, aes(x = TSNE1, y = TSNE2, color = HC_Cluster, shape = Expert_Diagnose)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("red", "green", "blue", "purple")) +
  scale_shape_manual(values = c(1, 2, 3, 4, 5, 6, 15, 17, 18, 19)) +
  theme_minimal() +
  labs(title = "t-SNE Projection with Hierarchical Clustering",
       x = "t-SNE Dimension 1",
       y = "t-SNE Dimension 2",
       color = "Hierarchical Cluster",
       shape = "Expert Diagnose")

# Calculate the Silhouette score for the hierarchical clusters
sil_scores_hc_tsne <- silhouette(clusters_hc_tsne, dist_tsne)
plot(sil_scores_hc_tsne, main = "Silhouette Plot for Hierarchical Clustering")
cat("Average silhouette width (Hierarchical):", summary(sil_scores_hc_tsne)$avg.width, "\n")











