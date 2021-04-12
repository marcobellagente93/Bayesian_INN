# Bayesian\_INN
This code uses the FrEIA framework to build a normalizing flow with several options, including:
* Adversarial training
* Bayesian linear layers
* MMD loss

Brief description of the content:
* train is the main piece of code
* FrEIA is directly cloned from a public repository from the local computer science group working on normalizing flows. It is imported and used in model.py;
* model is where the INN class is implemented. It contains the BlockConstructor class which is needed for using FrEIA classes, and the INN class. Note that BlockConstructor is where we use regular or bayesian linear layers;
* gan\_models contain several implementations of discriminator and generators;
* problayer contains the implementation of the variational linear layers;
* load\_data contains functions related to loading and preprocessing of data using in training, plus another couple of useful functions which shouldn't be there;
* utils is a folder containing a bunch of stuff. The plotting subfolder contains everything used for making plots, while train\_utils contains many functions used in train, such as log printings, saving and loading checkpoints, and the functions used to run operations on the 4-vectors (such as setting particles on-shell);
* coupling\_layer contains the implementation of the All-In-One coupling layer which is used in this particular implementation of RealNVP;
* losses contains the implementation of the MMD loss which can be used during training.
