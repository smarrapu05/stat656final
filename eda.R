rm(list=ls())
library(igraph)
library(dplyr)
library(tidyr)
setwd('C:/Users/Saketh/Documents/STAT656/PROJECT/ENZYMES')

# Load dataset values from directory

edges <- read.csv('ENZYMES_A.txt',header = FALSE)
graphInd <- read.csv('ENZYMES_graph_indicator.txt',header=FALSE)$V1
graphLab <- read.csv('ENZYMES_graph_labels.txt',header=FALSE)$V1
nodeLab <- read.csv('ENZYMES_node_labels.txt',header=FALSE)$V1
nodeAtb <- read.csv('ENZYMES_node_attributes.txt',header=FALSE)
ngraph <- length(graphLab)
enzGraphs <- list()

#Build graphs into enzGraphs (list of graphs)

for (g_id in 1:ngraph){
  nodeId <- which(graphInd == g_id)
  gEdg <- edges[edges$V1 %in% nodeId,]
  mapping <- setNames(1:length(nodeId),nodeId)
  mappedEdg <- data.frame(
    V1 = mapping[as.character(gEdg$V1)],
    V2 = mapping[as.character(gEdg$V2)]
  )
  g <- graph_from_data_frame(mappedEdg, directed = FALSE, 
                             vertices = data.frame(name = 1:length(nodeId)))
  V(g)$sec_structure <- nodeLab[nodeId]
  for (col in 1:ncol(nodeAtb)) {
    g <- set_vertex_attr(g, paste0("attr_", col), value = nodeAtb[nodeId, col])
  }
  g$ec_class <- graphLab[g_id]
  enzGraphs[[g_id]] <- g
}
length(enzGraphs)

# Load necessary libraries
library(igraph)
library(dplyr)
library(tidyr)

# Initialize an empty data frame to store metrics for each individual graph
graph_metrics <- data.frame(
  graph_id = integer(),
  ec_class = integer(),
  node_count = numeric(),
  edge_count = numeric(),
  radius = numeric(),
  diameter = numeric(),
  mean_eigen_cent = numeric()
)

# Loop through the enzGraphs list to calculate metrics per graph
# (This assumes the `enzGraphs` list from the previous script is loaded)
for (i in 1:length(enzGraphs)) {
  g <- enzGraphs[[i]]
  
  # Basic Counts
  n_nodes <- vcount(g)
  n_edges <- ecount(g)
  
  # Structural Distances
  # unconnected=TRUE ensures it returns a valid number even if the graph has isolated nodes
  g_diam <- diameter(g, directed=FALSE, unconnected=TRUE)
  
  
  ecc <- eccentricity(g)
  valid_ecc <- ecc[ecc > 0]
  g_rad <- ifelse(length(valid_ecc) > 0, min(valid_ecc), NA)
  
  # Eigenvector Centrality 
  ev_cent <- eigen_centrality(g, directed=FALSE, scale=TRUE)$vector
  mean_ev <- mean(ev_cent, na.rm=TRUE)
  
  # Append the results for this graph to our dataframe
  graph_metrics <- rbind(graph_metrics, data.frame(
    graph_id = i,
    ec_class = g$ec_class,
    node_count = n_nodes,
    edge_count = n_edges,
    radius = g_rad,
    diameter = g_diam,
    mean_eigen_cent = mean_ev
  ))
}

# Aggregation: Global Statistics
global_summary <- graph_metrics %>%
  summarise(
    level = "Global Dataset",
    across(c(node_count, edge_count, radius, diameter, mean_eigen_cent),
           list(
             mean = ~mean(.x, na.rm = TRUE),
             median = ~median(.x, na.rm = TRUE),
             sd = ~sd(.x, na.rm = TRUE)
           ),
           .names = "{.col}_{.fn}")
  )

print("=== GLOBAL AGGREGATION ===")
# Transpose for easier reading in the console
print(t(global_summary))

#Aggregation: By Enzyme Commission (EC) Class
class_summary <- graph_metrics %>%
  group_by(ec_class) %>%
  summarise(
    across(c(node_count, edge_count, radius, diameter, mean_eigen_cent),
           list(
             mean = ~mean(.x, na.rm = TRUE),
             median = ~median(.x, na.rm = TRUE),
             sd = ~sd(.x, na.rm = TRUE)
           ),
           .names = "{.col}_{.fn}")
  )

print("=== AGGREGATION BY EC CLASS ===")
print(class_summary)

global_summary <- graph_metrics %>%
  summarise(
    ec_class = "global", # Set the class to "global"
    across(c(node_count, edge_count, radius, diameter, mean_eigen_cent),
           list(
             mean = ~mean(.x, na.rm = TRUE),
             median = ~median(.x, na.rm = TRUE),
             sd = ~sd(.x, na.rm = TRUE)
           ),
           .names = "{.col}_{.fn}")
  )

#Re-create the class summary and ensure ec_class is a character
class_summary <- graph_metrics %>%
  group_by(ec_class) %>%
  summarise(
    across(c(node_count, edge_count, radius, diameter, mean_eigen_cent),
           list(
             mean = ~mean(.x, na.rm = TRUE),
             median = ~median(.x, na.rm = TRUE),
             sd = ~sd(.x, na.rm = TRUE)
           ),
           .names = "{.col}_{.fn}")
  ) %>%
  mutate(ec_class = as.character(ec_class)) # Convert 1-6 to text to match "global"

#Combine the two dataframes using bind_rows
combined_summary <- bind_rows(global_summary, class_summary)

# Print to console to verify
print("=== COMBINED SUMMARY ===")
print(combined_summary)

#Export to a CSV file
write.csv(combined_summary, "enzyme_metrics_combined_summary.csv", row.names = FALSE)
print("Successfully exported to 'enzyme_metrics_combined_summary.csv'")

library(ggplot2)
library(patchwork) # For combining plots

# Ensure EC class is a factor for better plotting
graph_metrics$ec_class <- as.factor(graph_metrics$ec_class)

# Boxplots for all metrics
p1 <- ggplot(graph_metrics, aes(x=ec_class, y=node_count, fill=ec_class)) + 
  geom_boxplot() + theme_minimal() + labs(title="Node Count") + guides(fill="none")

p2 <- ggplot(graph_metrics, aes(x=ec_class, y=edge_count, fill=ec_class)) + 
  geom_boxplot() + theme_minimal() + labs(title="Edge Count") + guides(fill="none")

p3 <- ggplot(graph_metrics, aes(x=ec_class, y=mean_eigen_cent, fill=ec_class)) + 
  geom_boxplot() + theme_minimal() + labs(title="Mean Eigenvector Centrality") + guides(fill="none")

p4 <- ggplot(graph_metrics, aes(x=ec_class, y=diameter, fill=ec_class)) + 
  geom_boxplot() + theme_minimal() + labs(title="Diameter") + guides(fill="none")

# Combine the plots into a grid
(p1 | p2) / (p3 | p4) + plot_annotation(title = "Structural Metric Differences by Enzyme Class")

# Scatter Plot: Node Count vs Edge Count
ggplot(graph_metrics, aes(x=node_count, y=edge_count, color=ec_class)) +
  geom_point(alpha=0.6) +
  geom_smooth(method="lm", se=FALSE) +
  theme_minimal() +
  labs(title="Structural Scaling: Nodes vs Edges by Class",
       x="Number of Nodes (Residues)", y="Number of Edges (Contacts)")

class_means <- graph_metrics %>%
  group_by(ec_class) %>%
  summarise(
    node_count = mean(node_count, na.rm = TRUE),
    edge_count = mean(edge_count, na.rm = TRUE),
    radius = mean(radius, na.rm = TRUE),
    diameter = mean(diameter, na.rm = TRUE),
    mean_eigen_cent = mean(mean_eigen_cent, na.rm = TRUE)
  )

# Normalize the data (Min-Max Scaling) 
# This ensures that for each metric, the best class is 1 and worst is 0
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

class_means_norm <- class_means %>%
  mutate(across(-ec_class, normalize))

# Reshape data to "Long" format for ggplot
# This turns columns into rows: | ec_class | metric_name | normalized_value |
heatmap_data <- class_means_norm %>%
  pivot_longer(cols = -ec_class, names_to = "metric", values_to = "norm_value")

# Create the Heatmap
ggplot(heatmap_data, aes(x = factor(ec_class), y = metric, fill = norm_value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "yellow", high = "salmon", name = "Relative Scale") +
  geom_text(aes(label = round(norm_value, 2)), color = "black", size = 3) +
  theme_minimal() +
  labs(
    title = "Relative Structural Differences by Enzyme Class",
    subtitle = "Values are Min-Max Normalized (0 to 1) per Metric",
    x = "EC Class Number",
    y = "Structural Metric"
  ) +
  theme(
    axis.text.x = element_text(angle = 0),
    panel.grid = element_blank()
  )
# Export the raw graph-level metrics (600 rows) for the Machine Learning model
write.csv(graph_metrics, "enzyme_graph_metrics.csv", row.names = FALSE)

print("Successfully exported raw metrics to 'enzyme_graph_metrics.csv'")

metrics_to_test <- c("node_count", "edge_count", "radius", "diameter", "mean_eigen_cent")

lapply(metrics_to_test, function(m) {
  cat("\n--- ANOVA for:", m, "---\n")
  fit <- aov(as.formula(paste(m, "~ as.factor(ec_class)")), data = graph_metrics)
  print(summary(fit))
  
  # Only print Tukey if ANOVA is significant
  if(summary(fit)[[1]][["Pr(>F)"]][1] < 0.05) {
    cat("\nRunning Tukey\n")
    print(TukeyHSD(fit))
  }
})
