
######################
# Choose dataset: #
######################

dataset = '2d_linear'

#########
# Data: #
#########

weighted = 0
on_shell = 0
mom_cons = 0
scaler   = 1.

##############
# Training:  #
##############

lr = 1e-3
batch_size = 512
gamma = 0.999
weight_decay = 1e-5
betas = (0.9, 0.999)

do_rev = False
do_fwd = True

n_epochs = 1
n_its_per_epoch = 1

adversarial = False
mmd         = False
variational = False

#################
# Architecture: #
#################

# For cond. generation:
n_blocks = 2
n_units = 16
n_layers = 2

fc_dropout = 0.0

####################
# Logging/preview: #
####################

loss_names = ['L', 'L_rev']
progress_bar = True                         # Show a progress bar of each epoch

show_interval = 5
save_interval = 10

###################
# Loading/saving: #
###################

test = False
train = True
predict = 'uniform'

save_model = True
load_model = False

save_dir = './experiments'
checkpoint_on_error = False
