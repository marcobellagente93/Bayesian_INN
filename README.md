# Bayesian\_INN
This code uses the FrEIA framework to build a normalizing flow with several options, including:
* Bayesian linear layers
* MMD loss

Brief description of the content:
* parameters are passed via the config file;
* train is the function to call to perform the training loop and save results/checkpoints;
* FrEIA is directly cloned from the corresponding public repository. I'll add the link at some point. It's necessary to use the reversible\_graph class, plus some nice functions like permute\_random are already implemented there;
* model is where the INN class is implemented. It contains the BlockConstructor class which is needed for using FrEIA classes, and the INN class. Note that BlockConstructor is where we introduce either regular or bayesian linear layers;
* problayer contains the implementation of the variational linear layers. Options will probably be adjusted/modified later;
* load\_data contains functions related to loading and preprocessing of data using in training;
* utils is a folder containing a bunch of stuff. The plotting subfolder contains everything used for making plots, while train\_utils contains many functions used in train, such as log printings, saving and loading checkpoints, and the functions used to run operations on the 4-vectors (such as setting particles on-shell);
* coupling\_layer contains the implementation of the All-In-One coupling layer which is used in this particular implementation of RealNVP;
* losses contains the implementation of the MMD loss which can be used during training.
