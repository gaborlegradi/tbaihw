# %% 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print(f'tf.__version__: {tf.__version__}\n'
      f'tf.config.list_physical_devices("GPU"):\n'
      f'{tf.config.list_physical_devices("GPU")}', sep='')
from utils import load_and_preprocess_data
from vaec import VAEC

# %%

x_train, y_train, x_test, y_test, mean_x_train, dev_x_train = \
    load_and_preprocess_data(split_ratio=0.7)

# %%

config= {
    'global_input_dim' : 30,
    'encoder_layers' : [16, 8],
    'decoder_layers' : [8, 16],
    'latent_dim' : 2,
    'classifier_layers' : [8, 4], # [8, 4], # [],
    'activation': 'elu',
    }

# %%

vaec = VAEC(config=config, 
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, 
            log_mark='_LD02_var_mlp', summary=False)

"""vaec.fit_var_mlp_classifier(100)

pred = vaec.var_mlp_classifier.predict(x_train)
print(pred[0][:7])
print(pred[1][:7])
print(pred[2][:7])
print(np.exp(pred[2][:7]/2))
"""
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




# %%
vaec.fit_var_mlp_classifier(10)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(20)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(40)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(80)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(160)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(320)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(370)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(1000)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(1000)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(1000)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(1000)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(1000)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(1000)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(1000)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(1000)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
vaec.fit_var_mlp_classifier(1000)
pred_train = vaec.var_mlp_classifier.predict(x_train)
pred_test = vaec.var_mlp_classifier.predict(x_test)
# %%
plot_label_clusters(pred_train, y_train, pred_test, y_test)
# %%
