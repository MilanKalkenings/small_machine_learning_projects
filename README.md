# Transfer Learning with U-Net
- implementation of a (small) unet architecture
    - U-Nets have two main components:
        - **down:** spatial resolution $\downarrow$, channel resolution $\uparrow$. Creates dense input representation
        - **up:** spatial resolution $\uparrow$, channel resolution $\downarrow$. Output is often of same (spatial) resolution as down-input.
    - skip-connections (concatenation) between up and down blocks of same resolution improve gradient flow to early layers
- pretraining of the down part with image classification using a classification head
- fine tuning on image segmentation data in two stages:
  1. adjusting upwards part with frozen pretrained downwards part
  2. end-to-end fine tuning of the downwards part and the upwards part
 
# MIMO Classification
- implementation of a MIMO (Multi-Input Multi-Output) Ensemble
    - implicit ensemble that learns independent subnetworks within one neural network
    - exploits network capacity
    - M ensemble predictions with a single forward pass
    - few more time and space complexity (less than 1%), but can converge in independent subnetworks with decorrelated errors/high disagreement
    - M ensemble predictions allow uncertainty measure
    - MIMO paper: https://openreview.net/pdf?id=OGg9XnKxFAH
- cifar10 preprocessing for MIMO ensembles

# Monte Carlo Dropout 
- implementation of a monte carlo dropout CNN on MNIST
    - drops out certain activations not only during training but also in inference
    - multiple forward passes create ensemble predictions that can be averaged to increase the generalization ability
    - mc dropout paper: https://arxiv.org/pdf/1506.02142.pdf
- performed 10 runs to compare monte carlo ensembles of different size with a normal dropout baseline
![mcd_accuracies](https://user-images.githubusercontent.com/70267800/213738311-6a15cfe2-e859-4809-aad2-c9d925c783b4.png)

# Cotton Embeddings with TripletMarginLoss
- implementation of a multihead resnet
- classification head classifies cotton plants (healthy, powdy mildew, aphids, army worm, bacterial blight, target spot)
- embedding head creates 2d latent space using the TripletMarginLoss on triplets of data points:
    - $a$ = anchor (embedding of a data point)
    - $p$ = positive (embedding of a data point of same class as $a$)
    - $n$ = negative (embedding of a data point of different class as $a$)
    - $L(a, p, n) = max(d(a, p) - d(a, n) + \alpha, 0)$ (with $d$ being a distance, \alpha a desired margin)
    - learns to fulfill $d(a, p) + \alpha < d(a, n)$

<img width="121" alt="triplet" src="https://user-images.githubusercontent.com/70267800/217182040-84d16b43-a3fb-46d7-95f3-d400739c1b88.png">

- hard triplet mining
    - find $p$ so that $p$ is the **most different** embedding to $a$ of **same class**
    - find $n$ so that $n$ is the **most similar** embedding to $a$ of **different class**

![results_debug](https://user-images.githubusercontent.com/70267800/217192469-80f763b7-78fc-4b50-8052-f558b8d64971.png)


# BERT MLM
- fine tuning of a pretrained bert model with masked language modeling (mlm)
    - texts are split into tokens ((sub-) words)
    - each token is masked with a certain probability (usually 15%)
    - model "fills the gaps" with tokens (simple classification to check if the predicted token is correct)
- data set: [turkey and syria earthquake tweets](https://www.kaggle.com/datasets/swaptr/turkey-earthquake-tweets)


# Carlini-Wagner Attack
- resnet trained on image net
- carlini wagner attack to create adversarial example of a gold fish to be misclassified as a flamingo
    - target class $t$: flamingo
    - change $x$ using gradient descent so that its predicted class probability is at least $\kappa$ bigger than that of the second most likely class
    - makes $x$ and $x_0$ more similar to each other, if the softmax output is of the desired form
    - carlini wagner criterion: $max(-\kappa, \underset{j\neq t}{max}(p_j)-p_t) + ||x-x_0||^2_2$

![result](https://user-images.githubusercontent.com/70267800/222924302-8e901e6d-092e-43d3-a249-1b3f8a269982.png)

# Faster R-CNN Object Detection
- detecting litter objects on forest floor
- created data set
    - made photos of forest floor
    - most photos contain at least one litter object (plastic, metal, paper, glass)
    - annotated litter objects with bounding boxes (corner coordinates)
    - photos contain benign confounders, i.e. natural objects that are easily confused with litter (reflecting puddles, colorful blossoms and berries, ...)
    - annotated data is available on https://www.kaggle.com/datasets/milankalkenings/litter-on-forest-floor-object-detection
- fine tuned Faster R-CNN (pretrained on COCO)

![all](https://user-images.githubusercontent.com/70267800/226890943-c7033e17-96a2-4cd2-b8e3-b496267a3bd3.png)


# Self Training
- given: few labeled data points, many unlabeled data points (from approx. same distribution)
- iteratively add semi-supervised labels to the unlabeled data points
    1. train model on labeled training set
    2. predict labels of unlabeled data points
    3. add data point(s) with most confident prediction to labeled training set
    4. repeat 1. to 3. until no improvement is achieved on validation data
  
![accs](https://user-images.githubusercontent.com/70267800/236207235-c03f8263-1805-42b0-b03e-b9c4edc17117.png)

# Semi-Supervised Clustering (Cluster-Then-Label)
- highly related to k-means, but with supervised guidance
- given: few labeled data points, many unlabeled data points (from approx. same distribution)
- perform clustering, align clusters to respect class labels, and predict according to the majority class per cluster
    - objective function = mean distance to cluster center + $\alpha$ class impurity per cluster
    1. initiate cluster centers
    2. unsupervised: assign data points to closest cluster center
    3. supervised: move labeled data point to other cluster, if that minimizes objective function
    4. cluster_center = mean(cluster)
    5. repeat 2. - 4. until no improvement in objective fnction is achieved
    
![results](https://user-images.githubusercontent.com/70267800/236207413-5b3ca6a7-4a30-48aa-a4da-e3e4aa04b0a3.png)


# Autoencoder
- training an autoencoder on MNIST and CIFAR100
    - if autoencoder is trained to reconstruct instances of data set X, it is likely to achieve good results on reconstructing instances of data set Y, if X and Y are similar enough. 
    - pretraining an encoder within an autoencoder, and later using it for as a feature learner in a classifier can speed up the training process, because the encoder already learned how to extract general features in the given data
    - a well trained autoencoder can be used to generate new data points that still contain the data signal, but add further noise (similar to data augmentation)

![mnist_prediction](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/57d70258-c6c0-4cb6-ad52-e95729fadb1c)

# Positional Encoding
- visualization of positional encodings as used in Transformers

![pe](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/379a8d22-0a63-4a74-8fcf-937d84c8702f)

