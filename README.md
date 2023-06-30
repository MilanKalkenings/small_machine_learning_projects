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


# learning rate range test
- implementation of learning rate range test (lrrt)
- stable algorithm for determining learning rates (and other hyperparameters) in deep learning projects
- naive comparison between initial and last batch loss can fail to detect best lr due to variance in the batch losses

1. define a set of lr candidates
2. train from the same checkpoint on few batches with each lr candidate
3. fit a line through the batch losses for each lr candidate
4. return the lr candidate with the steepest negative line slope

![lrrt](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/819fa710-5619-424d-9cf7-53385b7abbd8)

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

# MIMO Classification
- implementation of a MIMO (Multi-Input Multi-Output) Ensemble
    - implicit ensemble that learns independent subnetworks within one neural network
    - exploits network capacity
    - M ensemble predictions with a single forward pass
    - few more time and space complexity (less than 1%), but can converge in independent subnetworks with decorrelated errors/high disagreement
    - M ensemble predictions allow uncertainty measure
    - MIMO paper: https://openreview.net/pdf?id=OGg9XnKxFAH
- cifar10 preprocessing for MIMO ensembles
- my presentation slides about the MIMO paper
- my seminar paper reviewing the MIMO paper

# Monte Carlo Dropout 
- implementation of a monte carlo dropout CNN on MNIST
    - drops out certain activations not only during training but also in inference
    - multiple forward passes create ensemble predictions that can be averaged to increase the generalization ability
    - mc dropout paper: https://arxiv.org/pdf/1506.02142.pdf
- performed 10 runs to compare monte carlo ensembles of different size with a normal dropout baseline
![mcd_accuracies](https://user-images.githubusercontent.com/70267800/213738311-6a15cfe2-e859-4809-aad2-c9d925c783b4.png)

# BERT MLM
- fine tuning of a pretrained bert model with masked language modeling (mlm)
    - texts are split into tokens ((sub-) words)
    - each token is masked with a certain probability (usually 15%)
    - model "fills the gaps" with tokens (simple classification to check if the predicted token is correct)
- data set: [turkey and syria earthquake tweets](https://www.kaggle.com/datasets/swaptr/turkey-earthquake-tweets)


# Adversarial Attacks
- carlini wagner attack (targeted attack)
    - target class $t$: flamingo
    - change $x$ using gradient descent so that target probability is at least $\kappa$ bigger than second biggest probability
    - makes $x$ and $x_0$ more similar to each other, if softmax output is of desired form
    - carlini wagner criterion: $max(-\kappa, \underset{j\neq t}{max}(p_j)-p_t) + ||x-x_0||^2_2$

![result](https://user-images.githubusercontent.com/70267800/222924302-8e901e6d-092e-43d3-a249-1b3f8a269982.png)

- fast gradient sign method (untargeted attack)
    - goal: create $x_{fgsm}$ that is close to $x$ and leads to misclassification
    - $x_{fgsm}=x - sign(\frac{\partial f(x)_{y}}{\partial x})) \cdot \epsilon$
    - $sign(\frac{\partial f(x)_{y}}{\partial x})):$ direction in which score for class $y$, increases
    - strong perturbations can make $x_{fgsm}$ OOD and can lead to even higher class score, because gradient is only local approximation 

![fgsm](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/8ecc4ac7-fa2c-4bc0-b38a-9a885ce7f5c3)



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

# Unsupervised Classification Support
- semi supervised training with cross entropy and unsupervised support
- unsupervised support: loss functions that can be calculated on unlabeled data points
- **stability loss**: favors similar predictions for $n$ augmented versions of same data point
    - risk: trivial solution is to always predict the same vector
- **mutual exclusivity loss**: favors to low-entropy softmax outputs
    - leads to decision boundary through low-density regions in feature space
    - prevents trivial solution for stability loss

<img width="358" alt="table" src="https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/32f75c6f-aadc-4313-99d6-8daa5376fca2">


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

# Self Training
- given: few labeled data points, many unlabeled data points (from approx. same distribution)
- iteratively add semi-supervised labels to the unlabeled data points
    1. train model on labeled training set
    2. predict labels of unlabeled data points
    3. add data point(s) with most confident prediction to labeled training set
    4. repeat 1. to 3. until no improvement is achieved on validation data
  
![accs](https://user-images.githubusercontent.com/70267800/236207235-c03f8263-1805-42b0-b03e-b9c4edc17117.png)

# Autoencoder
- training an autoencoder on MNIST and CIFAR100
    - if autoencoder is trained to reconstruct instances of data set X, it is likely to achieve good results on reconstructing instances of data set Y, if X and Y are similar enough. 
    - pretraining an encoder within an autoencoder, and later using it for as a feature learner in a classifier can speed up the training process, because the encoder already learned how to extract general features in the given data
    - a well trained autoencoder can be used to generate new data points that still contain the data signal, but add further noise (similar to data augmentation)
- training a variational autoencoder on MNIST
    - learns reparameterization
        - $encoding=\mu + (\sigma \epsilon), \epsilon \textasciitilde \mathcal{N}(0,1)$
        - $\mu=linear(encoding_{raw})$
        - $\sigma=exp(linear(encoding_{raw}))$
    - visualization of PCA-reduced latent space
    - visualization of multiple interpolation stages between two instances in latent space

![mnist_prediction](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/57d70258-c6c0-4cb6-ad52-e95729fadb1c)

# Fake News Feature Importance
- simple feature engineering with tabular data
    - number of words (title and body)
    - number of exclamation marks (title and body)
    - number of question marks (title and body)
    - lexical diversity (title and body)
    - $\frac{\text{number of title words}}{\text{number of title words} + \text{number of body words}}$
- feature importance evaluation with random forest
- good resuls can already be achieved by using only 1 feature
- further sklearn mechanics used (stacking ensemble, gridsearch)

![f1](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/acc58d9c-1ec0-4be0-812c-9a758eca1aa5)


# Meta Learning
- meta learning with reptile:
    1. sample a task from a set of tasks
    2. for few iterations, update model parameters on sampled task
    3. repeat 1. and 2. until model performs well over all tasks
- reptile pretraining improves one shot results
    - 31% one shot test accuracy without reptile pretraining
    - 84% one shot test accuracy with reptile pretraining
  
<img width="557" alt="reptile" src="https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/e9af08e0-64ee-4d98-883c-657ed29001eb">


# Positional Encoding
- visualization of positional encodings as used in Transformers

![pe](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/379a8d22-0a63-4a74-8fcf-937d84c8702f)

# cam (+ smoothgrad + guided)
- implementation of regular class activation maps (cam) 
    - cams show the importance of the individual input elements for the activation for the respective class
    - cams are based on the gradient of a class activation w.r.t. the input $\frac{\partial f_\theta(x)_{class}}{\partial x}$
    - cams can e.g. be used for
      - inferring weakly supervised labels (here: bounding boxes from classification labels)
      - model debugging
      - deducting model design decisions
- implementation of smoothgrad
    - better cams by averaging over gradients for $n$ noisy versions of the input
- implementation of guided cam
    - better cams by only propagating positive gradients back
- gradually masking out one object decreases its class score
    
![benchmark](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/160549b5-ae37-4e45-b4f7-0d848f31271d)

![masking](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/d499bd84-ecbb-45b3-83a5-a6f8319a21d0)

![probs](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/21416f03-d0ff-48d4-bb42-3164dff3824c)


# Consistency Regularization
- unlabeled data points still belong to the same category as perturbated versions of themselves
- regularizer $\lambda ||f(x) - f(x_{augmented})||^2_2$ forces the model to predict similar outputs for $x$ and $x_{augmented}$ 

![vary_lambda](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/75c0e4dd-c9c9-484f-8135-8d86b0fc305e)

# Multiple Instance Learning
- weakly supervised method: 
    - data instances come in bags 
    - only the labels of few instances are known
    - predictions are made on bag level
- binary classification: bag is of positive class, if it contains a "2", else bag is of negative class
- attention pooling can be used to deduct instance level predictions
    - model pays high attention to instances of positive class (even to those that are not labeled)
- mil-attention-pooling paper: http://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf
![pos](https://github.com/MilanKalkenings/small_machine_learning_projects/assets/70267800/27151a89-dda6-4a8f-9e8c-ad3c155b38d3)

