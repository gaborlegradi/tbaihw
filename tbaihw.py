# %% [markdown]
# Imports:
# - Essentials like MatPlotLib PyPlot, Numpy and TensorFlow itself.
# - Data load+preprocess tool from utils.
# - Variational AutoEncoder + Classifier (VAEC) class.
# 
# Checking TensorFlow version.
# %% 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print(f'tf.__version__: {tf.__version__}\n'
      f'tf.config.list_physical_devices("GPU"):\n'
      f'{tf.config.list_physical_devices("GPU")}', sep='')
from utils import load_and_preprocess_data
from vaec import VAEC

# %% [markdown]
# Calling load_and_preprocess_data:
# - Load data.
# - Normalize.
# - Shuffle with random seed and split.
# 
# Gives back:
# - Train and test data, and 
# - normalization vectors which are needed for preprocessing a possible out-of databease input.
# %%

x_train, y_train, x_test, y_test, mean_x_train, dev_x_train = \
    load_and_preprocess_data(split_ratio=0.7)

# %% [markdown]
# Config:
# - Encoder contains 2 layers with 16 and 8 neurons.
# - Decoder layer is now irrelevant, not affected by training in this test.
# - latent_dim = 2.
# - No classifier MLP hidden layers, final single sigmoid neuron is directly 
# connected to neck of latent_dim dimension.
# - ELU activation is used.
# %%

config= {
    'global_input_dim' : 30,
    'encoder_layers' : [16, 8],
    'decoder_layers' : [8, 16],
    'latent_dim' : 2,
    'classifier_layers' : [], # [8, 4], # [],
    'activation': 'elu',
    }

# %% [markdown]
# Instantiating Variational AutoEncoder + Classifier (VAEC) class.
# %%
vaec = VAEC(config=config, 
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, 
            log_mark='_LD02_var_mlp', summary=False)

# %% [markdown]
# Definition of ploting function.
# %%

def plot_label_clusters(pred_train, y_train, pred_test, y_test):
    
    plt.figure(figsize=(20, 30))

    plt.subplot(321)
    plt.scatter(pred_train[3][:, 0], pred_train[3][:, 1], c=y_train)
    plt.title("Train::z")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    plt.subplot(322)
    plt.scatter(pred_test[3][:, 0], pred_test[3][:, 1], c=y_test)
    plt.title("Test::z")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    plt.subplot(323)
    plt.scatter(pred_train[1][:, 0], pred_train[1][:, 1], c=y_train)
    plt.title("Train::z_mean")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.subplot(324)
    plt.scatter(pred_test[1][:, 0], pred_test[1][:, 1], c=y_test)
    plt.title("Test::z_mean")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.subplot(325)
    plt.scatter(pred_train[1][:, 0], pred_train[1][:, 1], c=y_train)
    plt.title("Train::z_mean")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")

    plt.subplot(326)
    plt.scatter(pred_test[1][:, 0], pred_test[1][:, 1], c=y_test)
    plt.title("Test::z_mean")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

# %% [markdown]
# Training: epoch 1-8
# %%
vaec.fit_var_mlp_classifier(8, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 9-16
# %%
vaec.fit_var_mlp_classifier(8, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 17-32
# %%
vaec.fit_var_mlp_classifier(16, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 33-64
# %%
vaec.fit_var_mlp_classifier(32, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 65-128
# %%
vaec.fit_var_mlp_classifier(64, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 129-256
# %%
vaec.fit_var_mlp_classifier(128, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 257-512
# %%
vaec.fit_var_mlp_classifier(256, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 513-1000
# %%
vaec.fit_var_mlp_classifier(488, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 1001-2000
# %%
vaec.fit_var_mlp_classifier(1000, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 2001-3000
# %%
vaec.fit_var_mlp_classifier(1000, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 3001-4000
# %%
vaec.fit_var_mlp_classifier(1000, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 4001-5000
# %%
vaec.fit_var_mlp_classifier(1000, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 5001-6000
# %%
vaec.fit_var_mlp_classifier(1000, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 6001-7000
# %%
vaec.fit_var_mlp_classifier(1000, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 7001-8000
# %%
vaec.fit_var_mlp_classifier(1000, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 8001-9000
# %%
vaec.fit_var_mlp_classifier(1000, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)

# %% [markdown]
# Training: epoch 9001-10000
# %%
vaec.fit_var_mlp_classifier(1000, verbose=0)
pred_train = vaec.var_mlp_classifier.predict(x_train, verbose=0)
pred_test = vaec.var_mlp_classifier.predict(x_test, verbose=0)
plot_label_clusters(pred_train, y_train, pred_test, y_test)
