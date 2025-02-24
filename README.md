# Rock-Image-Analysis-Classification-Clustering
Question
In this problem, you will use Rock dataset located here: https://osf.io/d6b9y to an external site.. 

1. Apply PCA to the images from folder '360 Rocks'. How many components do you need to preserve 95% of the variance? [3 points]
2. Plot 10 images of your choice in the original form (without PCA) and then plot their reconstruction (projection in the original space) after you kept 95% of variance using PCA. [3points]
3. Each of the images belongs to one of three rock categories. The category is indicated by the first letter in the filename (I, M and S). We will now try to see if the visualization can 
   help us identify different clusters.
  A. Use PCA to reduce dimensionality to only 2 dimensions. How much of the variance is explained with the first two principal components? [2 points].
  B. Plot a 2D scatter plot of the images spanned by the first two principal components. Each image will be represented with a dot. Make the color of the dot correspond to the image       
     category (so you will have three different colors). Then add some rock images to the visualization to better understand what features in the images are accounting for the majority of       variance in the data (your visualization should look similar to the one after line 71 in this file https://github.com/ageron/handson-ml3/blob/main/08_dimensionality_reduction.ipynb to an external site. but with images of rocks instead of MNIST digits). Repeat the process and create the same type of plots for 
     t-SNE, LLE and MDS. [6 points]
  D. Which of the visualizations do you prefer?[1 point]
4. Now let's see if these dimensionality reduction techniques can give us similar features to those that humans use to judge the images. File mds_360.txt contains 8 features for each of 
   the images (rankings are in the same order as the images in  '360 Rocks' folder. Run PCA, t-SNE, LLE and MDS to reduce the dimensionality of the images to 8. Then, compare those image      embeddings with the ones from humans that are in the mds_360.txt file. Use Procrustes analysis to do the comparison (here is one example of how to do that mtx1, mtx2, disparity =     
   procrustes(matrix_with_human_data, matrix_with_pca_embeddings_data). Here matrix_with_human_data and matrix_with_pca_embeddings_data should be 360 by 8. disparity will tell you the 
   difference in the data. Report disparity for each of the four dimensionality reduction methods. Compute the correlation coefficient between each dimension of mtx1 and mtx2 for each of 
   the four methods - display results in a table. [7 points]
5. Cluster the 360 images using K-Means.
  A. To speed up the algorithm, use PCA to reduce the dimensionality of the dataset to two. Determine the number of clusters using one of the techniques we discussed in class. [4 points]
  B. Visualize the clusters in a similar way to the visualization after line 28 here: https://github.com/ageron/handson-ml3/blob/main/08_dimensionality_reduction.ipynb 
  to an external          site., but color each dot based on the clusters it belongs to using the labels taken from the filename as in question 3  (I, M and S). [4 points]
6. Cluster the 360 images using EM.
  A. Same as in the previous question, to speed up the algorithm, use PCA to reduce the dimensionality of the dataset to two. Determine the number of clusters using one of the techniques 
     we discussed in class. [4 points]
  B. Visualize the clusters in a similar way to the visualization after line 28 here:
       https://github.com/ageron/handson-ml3/blob/main/09_unsupervised_learning.ipynb to an external   
     site., but color each dot based on the clusters it belongs to using the labels taken from the filename as in question 3  (I, M and S). [4 points]
  C. Use the model to generate 20 new rocks (using the sample() method), and visualize them in the original image space (since you used PCA, you will need to use its inverse_transform() 
     method).  [4 points]
8. Build a feedforward neural network (using dense and/or CNN layers) with a few hidden layers (we suggest using Keras (within Tensorflow) or Pytorch). Train the network to classify on 360    rock images using rock name as the label - the category is indicated by the first letter in the filename (I, M and S). Use images from '120 Rocks' folder as your validation data. Choose    the number of neurons you find appropriate and efficient (so you have enough time to run it), but make the last layer before the softmax should consist of 8 neurons. The hidden layers      should have ReLU activation function. Train the network for multiple epochs until it converges (if the process is too slow, tweak the learning rate and consider simplifying your         
   network). We will not deduct points based on the simplicity of your network, but we expect you to have performance that is above chance performance that could be obtained with an 
   untrained network - in other words, we expect to see train and validation loss decrease and accuracy increase throughout the training. We recommend using Colab (the free version should 
   be totally fine), but make sure to run it with a GPU to speed up the training - to add a GPU on Colab go to Edit->Notebook settings). 
  A. Report the training time (use code to do this). [1 point]
  B. Plot training and validation loss and accuracy as a function of training epochs. [13 points]
  C. How many parameters does the network have? How many of those parameters are bias parameters? [1 points]
  D. Compare the activity of neurons in the next to the last layer (the one with 8 neurons) with the human data. (to get human data use mds_360.txt and mds_120.txt files). Similar to     
     before, use Procrustes analysis to do the comparison.  For training and validation data (separately), report disparity and compute the correlation coefficient between each dimension 
     of mtx1 and mtx2. Display results in a table. [3 points]
