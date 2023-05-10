# collection of small machine learning projects

## unet transfer
- implementation of a (small) unet architecture
- pretraining of the downwards part with image classification
- fine tuning on image segmentation data in two stages:
  1. adjusting a unet upwards part with frozen pretrained downwards part
  2. end-to-end fine tuning of the downwards part and the upwards part
  
  
### unet
unets consist of two main components:
- **downwards:** each of its blocks decreases the spatial resolution and increases the channel dimension. creates a low-resolution representation of the input
- **upwards:** each of its blocks increases the spatial resolution and decreases the channel dimension. usually creates an output of the same dimensionality as the input to the downards component

unets have skip-connections between the downwards blocks and the upwards blocks to improve the gradient flow to the first layers during backprop.
the skip connections concatenate the outputs of downwards blocks with outputs of upwards blocks that are of the same dimensionality.

## mimo cls
- implementation of a MIMO (Multi-Input Multi-Output) Ensemble
- cifar10 data loading for MIMO ensembles

### mimo
- implicit ensemble that learns independent subnetworks within one neural network
- exploits network capacity
- M ensemble predictions with a single forward pass
- few more time and space complexity (less than 1%), but can converge in independent subnetworks with decorrelated errors/high disagreement
- M ensemble predictions allow uncertainty measure
- MIMO paper: https://openreview.net/pdf?id=OGg9XnKxFAH


## mc dropout 
- implementation of a monte carlo dropout CNN on MNIST
- mc dropout drops out certain activations not only during training but also in inference. Multiple forward passes create ensemble predictions that can be averaged to increase the generalization ability
- performed 10 runs to compare monte carlo ensembles of different size with a normal dropout baseline
- mc dropout paper: https://arxiv.org/pdf/1506.02142.pdf
![mcd_accuracies](https://user-images.githubusercontent.com/70267800/213738311-6a15cfe2-e859-4809-aad2-c9d925c783b4.png)

## cotton TripletMarginLoss
- implementation of a multihead resnet
- cls head classifies cotton plants (healthy, powdy mildew, aphids, army worm, bacterial blight, target spot)
- emb head creates 2d latent space using the TripletMarginLoss
- hard triplet mining

![results_debug](https://user-images.githubusercontent.com/70267800/217192469-80f763b7-78fc-4b50-8052-f558b8d64971.png)


### triplet margin loss
- takes a triplet of embeddings:
  - $a$ = anchor (embedding of a data point)
  - $p$ = positive (embedding of a data point of same class as $a$)
  - $n$ = negative (embedding of a data point of different class as $a$)
- $L(a, p, n) = max(d(a, p) - d(a, n) + \alpha, 0)$ (with $d$ being a distance, \alpha a desired margin)
- learns to fulfill $d(a, p) + \alpha < d(a, n)$
<img width="121" alt="triplet" src="https://user-images.githubusercontent.com/70267800/217182040-84d16b43-a3fb-46d7-95f3-d400739c1b88.png">


### hard triplet mining
- find $p$ so that $p$ is the **most different** embedding to $a$ of **same class**
- find $n$ so that $n$ is the *most similar** embedding to $a$ of **different class**


## bert mlm
- fine tuning of a pretrained bert model with masked language modeling (mlm)
- data set: [turkey and syria earthquake tweets](https://www.kaggle.com/datasets/swaptr/turkey-earthquake-tweets)

### mlm
- texts are split into tokens ((sub-) words)
- each token is masked with a certain probability (usually 15%)
- model "fills the gaps" with tokens (simple classification to check if the predicted token is correct)


## carlini wagner
- resnet trained on image net
- carlini wagner attack to create an adversarial example of a gold fish to be misclassified as a flamingo
- target class $t$: flamingo
- change $x$ using gradient descent so that its predicted class probability is at least $\kappa$ bigger than that of the second most likely class
- makes $x$ and $x_0$ more similar to each other, if the softmax output is of the desired form
- carlini wagner criterion: $max(-\kappa, \underset{j\neq t}{max}(p_j)-p_t) + ||x-x_0||^2_2$

![result](https://user-images.githubusercontent.com/70267800/222924302-8e901e6d-092e-43d3-a249-1b3f8a269982.png)

## faster r-cnn object detection
- detecting litter objects on forest floor
- created data set
- fine tuned faster r-cnn pretrained on COCO

### dataset 
- made photos of forest floor
- most photos contain at least one litter object (plastic, metal, paper, glass)
- annotated litter objects with bounding boxes (corner coordinates)
- photos contain benign confounders, i.e. natural objects that are easily confused with litter (reflecting puddles, colorful blossoms and berries, ...)
- annotated data is available for free use on https://www.kaggle.com/datasets/milankalkenings/litter-on-forest-floor-object-detection

![all](https://user-images.githubusercontent.com/70267800/226890943-c7033e17-96a2-4cd2-b8e3-b496267a3bd3.png)


## self training / self learning
- given: few labeled data points, many unlabeled data points (from approx. same distribution)
- iteratively add semi-supervised labels to the unlabeled data points
  1. train the model on the labeled training set
  2. predict the labels of the unlabeled data points
  3. add the data point(s) with the most confident prediction to the labeled training set
  4. repeat 1. to 3. until no improvement is achieved on validation data
  
![accs](https://user-images.githubusercontent.com/70267800/236207235-c03f8263-1805-42b0-b03e-b9c4edc17117.png)

## semi-supervised clustering (cluster-then-label)
- highly related to k-means, but with supervised guidance
- given: few labeled data points, many unlabeled data points (from approx. same distribution)
- perform clustering, align clusters to respect class labels, and predict according to the majority class per cluster
  1. initiate cluster centers
  2. unsupervised: assign data points to closest cluster center
  3. supervised: move labeled data point to other cluster, if that minimizes objective function
  4. cluster_center = mean(cluster)
  5. repeat 2. - 4. until no improvement in objective fnction is achieved
- objective function = mean distance to cluster center + $\alpha$ class impurity per cluster

![results](https://user-images.githubusercontent.com/70267800/236207413-5b3ca6a7-4a30-48aa-a4da-e3e4aa04b0a3.png)


## autoencoder
- training an autoencoder on MNIST and CIFAR100
- if an autoencoder is trained to reconstruct instances of data set X, it is likely to achieve good results on reconstructing instances of data set Y, if X and Y are similar enough. 
- autoencoders can (to some degree) learn *general reconstruction abilities* on equal data types
- pretraining an *encoder* within an autoencoder, and later using it for as a feature learner in a classifier can speed up the training process, because the encoder already learned how to extract general features in the given data
- a well trained autoencoder can be used to generate new data points that still contain the data signal, but add further noise (similar to data augmentation)

![mnist_prediction](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/57d70258-c6c0-4cb6-ad52-e95729fadb1c)
