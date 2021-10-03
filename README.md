\#How to Animate a Neural Network 1. Declare a basic array 2. build a
dataframe defining every nodes as rows 3. Build a simple neural Network
in tensorflow and Keras 4. Stack dataframe geenrated per epoch 5. Draw
individual weights onto each connections and each epoch

``` r
rm(list = ls())

#no_nodes_per_layer <- c(1, 4, 3, 5, 2)
#layers <-  length(no_nodes_per_layer)
```

``` r
generate_data <- function(no_nodes_per_layer = c(2, 3, 2, 3, 2)) {
  
  layers <- length(no_nodes_per_layer)
  if(length(no_nodes_per_layer) < layers){
    no_nodes_per_layer <- c(no_nodes_per_layer,seq(1,layers-length((no_nodes_per_layer))))
    }
  df <- c()
  max_nodes <- max(no_nodes_per_layer)
  
  ##Creating layers
  for (i in 1:layers) {
    df$layers <- c(df$layers, rep(i, no_nodes_per_layer[i]))
  }
  
  ##Creating nodes
  for (i in 1:layers) {
    nodes_no <- no_nodes_per_layer[i]
    diff <- (max_nodes - nodes_no)/2
    df$nodes <- c(df$nodes,seq(1,nodes_no)+diff)
  }
  df$epoch <- rep(1, length(df$layers))
  df$sizes <- rep(0, length(df$layers))

  
  return(df)
}
```

``` r
array <- c(4,6,8,6,3)
layers <- length(array)
df <- generate_data(no_nodes_per_layer = array)
df <- as.data.frame(df)
```

``` r
###Create neural network
library(tensorflow)
```

    ## Warning: package 'tensorflow' was built under R version 4.1.1

``` r
library(keras)
```

    ## Warning: package 'keras' was built under R version 4.1.1

``` r
normalize <- function(x)
    {
    return((x- min(x)) /(max(x)-min(x)))
    }

copydata <- function(array,n)
    {
    for(i in 1:n){array <- rbind(array,array)}
    return(array)
}

train_data <- unlist(iris[,1:4])
dim(train_data) <- c(150,4)
train_data  <- train_data  %>% scale() %>% normalize() 
train_data <- train_data %>% copydata(4)

train_label <- as.factor(iris[,5]) 
train_label <- train_label %>% as.numeric() %>% as.matrix() 
train_label <- train_label - 1 

# Convert labels to categorical one-hot encoding
train_label <- to_categorical(train_label,num_classes = 3)

train_label <- train_label %>% copydata(4)

rm(model)
```

    ## Warning in rm(model): object 'model' not found

``` r
model <- keras_model_sequential() %>% 
  layer_dense(units = array[1], activation = "sigmoid", input_shape = c(4)) %>% 
  layer_dense(units = array[2], activation = "sigmoid") %>% 
  layer_dense(units = array[3], activation = "sigmoid") %>% 
  layer_dense(units = array[4], activation = "sigmoid") %>%
  layer_dense(units = 3, activation = "softmax")

summary(model)
```

    ## Model: "sequential"
    ## ________________________________________________________________________________
    ## Layer (type)                        Output Shape                    Param #     
    ## ================================================================================
    ## dense_4 (Dense)                     (None, 4)                       20          
    ## ________________________________________________________________________________
    ## dense_3 (Dense)                     (None, 6)                       30          
    ## ________________________________________________________________________________
    ## dense_2 (Dense)                     (None, 8)                       56          
    ## ________________________________________________________________________________
    ## dense_1 (Dense)                     (None, 6)                       54          
    ## ________________________________________________________________________________
    ## dense (Dense)                       (None, 3)                       21          
    ## ================================================================================
    ## Total params: 181
    ## Trainable params: 181
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

``` r
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

weight <- list()
bias <- list()
model_stat <- list()
Overall <- list()
acc <- list()

for(e in 1:30){

history <- model %>% 
  fit(
    x = train_data, y = train_label,
    epochs = 1,
    use_multiprocessing=TRUE,
    batch_size = 20
  )
 for (l in 1:length(array)){
   weight[[l]] <- as.matrix(model$layers[[l]]$weights[[1]])
   bias[[l]]   <- as.matrix(model$layers[[l]]$weights[[2]])
 }
  model_stat$weight <- weight
  model_stat$bias <- bias
  
  Overall[[e]] <- model_stat
  acc[[e]] <- model$history$history$accuracy
} 
```

``` r
df 
```

    ##    layers nodes epoch sizes
    ## 1       1   3.0     1     0
    ## 2       1   4.0     1     0
    ## 3       1   5.0     1     0
    ## 4       1   6.0     1     0
    ## 5       2   2.0     1     0
    ## 6       2   3.0     1     0
    ## 7       2   4.0     1     0
    ## 8       2   5.0     1     0
    ## 9       2   6.0     1     0
    ## 10      2   7.0     1     0
    ## 11      3   1.0     1     0
    ## 12      3   2.0     1     0
    ## 13      3   3.0     1     0
    ## 14      3   4.0     1     0
    ## 15      3   5.0     1     0
    ## 16      3   6.0     1     0
    ## 17      3   7.0     1     0
    ## 18      3   8.0     1     0
    ## 19      4   2.0     1     0
    ## 20      4   3.0     1     0
    ## 21      4   4.0     1     0
    ## 22      4   5.0     1     0
    ## 23      4   6.0     1     0
    ## 24      4   7.0     1     0
    ## 25      5   3.5     1     0
    ## 26      5   4.5     1     0
    ## 27      5   5.5     1     0

``` r
df3 <- df

for(p in 2:length(Overall)) {
  df2 <- df
  df2$epoch <- p
  
  for (l in 1:layers) {
    df2$sizes[df2$layers==l] <-  Overall[[p]]$bias[[l]]
  }
  df3 <- rbind(df3, df2)
}

# Increase size for better plotting
df3$sizes <- normalize(df3$sizes)
```

``` r
library(ggplot2)
library(gganimate)

for (k in 1:30) {
  df4 <- df3[df3$epoch==k,]
  ## Create base Canvas
  p <-
    ggplot(data = df4)  + geom_point(aes(x = layers, y = nodes, size = sizes*2,color="red")) +
    labs(x = "Layers",
         y = "Nodes") +
    ggtitle(paste0("Epoch:", k,"  ","Accurancy:",round(acc[[k]],2))) +
    theme_bw()+
    theme(legend.position = "none")
  
  ##function to add all arrows
  for (i in 1:(layers - 1)) {
    for (j in 1:(array[i])) {
      for (o in 1:(array[i + 1])) {
        #print(paste0("i-", i))
        #print(paste0("j-", j))
        #print(paste0("o-", o))
        
        x1 <- i
        y1 <- df4[df4$layers == i, ]$nodes[j]
        x2 <- i + 1
        y2 <- df4[df4$layers == (i + 1), ]$nodes[o]
        
        coor <- data.frame(x = c(x1, x2), y = c(y1, y2))
        weights <- Overall[[k]]$weight[[x1+1]][[j,o]]
        
        p <- p + geom_path(data = coor, aes(x = x, y = y), alpha = weights, size=0.5,color="blue")
      }
    }
  }
  print(p)
  ggsave(paste0("plot-",k,".jpg"),
            width = 1920,
            height = 1080,
            units = c("px"),
            dpi = 300)
}
```

![](cutom-MLP-Plot_files/figure-gfm/create%20plot-1.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-2.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-3.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-4.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-5.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-6.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-7.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-8.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-9.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-10.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-11.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-12.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-13.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-14.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-15.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-16.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-17.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-18.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-19.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-20.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-21.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-22.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-23.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-24.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-25.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-26.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-27.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-28.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-29.png)<!-- -->![](cutom-MLP-Plot_files/figure-gfm/create%20plot-30.png)<!-- -->
