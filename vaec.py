import datetime
import tensorflow as tf
from netutils import create_encoder, create_decoder, create_classifier_head
from vaec_trainer import VAEC_Trainer

class VAEC():
    """
    *Variational AutoEncoder + Classifier*
    The object created by instantiating this class contains encoder, decoder and classifier_head parts and controls its 
    training:
    - Creates encoder decoder and classifier_head according to config then compile them.
    - Creates uncompiled mlp_classifier and mlp_classifier_with_z:
        - mlp_classifier gets input and outputs class prediction. This is to be used when training was performed in 
          'mlp_classifier' mode by VAEC_Trainer.
        - mlp_classifier_with_z gets input and outputs class perdiction + z of latent_dim dimension. This is to be 
          used when training was performed in 'autoencoder' and 'mlp_classifier' modes, consecutively, OR in 'all' mode.
    - At instantiation creates TensorBoard callback with logname log_mark + timestamp.
    - For training in 'all', 'autoencoder', 'classifier_head' or 'mlp_classifier' modes, functions fit_all_parts(), 
      fit_autoencoder(), fit_classifier_head() and fit_mlp_classifier() can be used.
    """
    def __init__(self, config, x_train, y_train, x_test, y_test, log_mark='default', summary=False):
        # Essential params, data and config are hold within object.
        self.config = config
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epoch = 0
        self.initial_epoch = 0
        # Creating TensorBoard callback with specific logname: log_mark + timestamp.
        dt_init = datetime.datetime.now()
        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{log_mark}_{dt_init:%y_%m_%d__%H_%M_%S}/',  update_freq=1)

        #Creating encoder, decoder and classifier_head.
        self.encoder = create_encoder(
            config['global_input_dim'], config['encoder_layers'], config['latent_dim'], config['activation'], summary=summary)
        self.decoder = create_decoder(
            config['global_input_dim'], config['decoder_layers'], config['latent_dim'], config['activation'], summary=summary)
        self.classifier_head = create_classifier_head(
            config['latent_dim'], config['classifier_layers'], config['activation'], summary=summary)

        # Creating trainers for 'all', 'autoencoder', 'classifier_head' and 'mlp_classifier' modes.
        self.trainer_all_parts = VAEC_Trainer(
            encoder=self.encoder, decoder=self.decoder, classifier_head=self.classifier_head, train_what='all')
        self.trainer_all_parts.compile(optimizer=tf.keras.optimizers.Adam())

        self.trainer_autoencoder = VAEC_Trainer(
            encoder=self.encoder, decoder=self.decoder, classifier_head=self.classifier_head, train_what='autoencoder')
        self.trainer_autoencoder.compile(optimizer=tf.keras.optimizers.Adam())

        self.trainer_classifier_head = VAEC_Trainer(
            encoder=self.encoder, decoder=self.decoder, classifier_head=self.classifier_head, train_what='classifier_head')
        self.trainer_classifier_head.compile(optimizer=tf.keras.optimizers.Adam())

        self.trainer_mlp_classifier = VAEC_Trainer(
            encoder=self.encoder, decoder=self.decoder, classifier_head=self.classifier_head, train_what='mlp_classifier')
        self.trainer_mlp_classifier.compile(optimizer=tf.keras.optimizers.Adam())

        self.trainer_var_mlp_classifier = VAEC_Trainer(
            encoder=self.encoder, decoder=self.decoder, classifier_head=self.classifier_head, train_what='var_mlp_classifier')
        self.trainer_var_mlp_classifier.compile(optimizer=tf.keras.optimizers.Adam())

        # Creating mlp_classifier and var_mlp_classifier.
        input = tf.keras.Input(shape=(config['global_input_dim'], ))
        z_mean, z_log_var, z = self.encoder(input)
        output = self.classifier_head(z_mean)
        self.mlp_classifier = tf.keras.Model(input, output)

        input = tf.keras.Input(shape=(config['global_input_dim'], ))
        z_mean, z_log_var, z = self.encoder(input)
        output = self.classifier_head(z_mean)
        self.var_mlp_classifier = tf.keras.Model(input, [output, z_mean, z_log_var, z])

    # Training functions for 'all', 'autoencoder', 'classifier_head' and 'mlp_classifier' modes' trainers.
    def fit_all_parts(self, epochs, batch_size=128, **kwargs):
        self.trainer_all_parts.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), 
                            initial_epoch=self.epoch, epochs=self.epoch + epochs, 
                            batch_size=batch_size, callbacks=[self.tb_callback], **kwargs)
        self.epoch += epochs
    def fit_autoencoder(self, epochs, batch_size=128, **kwargs):
        self.trainer_autoencoder.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), 
                            initial_epoch=self.epoch, epochs=self.epoch + epochs, 
                            batch_size=batch_size, callbacks=[self.tb_callback], **kwargs)
        self.epoch += epochs
    def fit_classifier_head(self, epochs, batch_size=128, **kwargs):
        self.trainer_classifier_head.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), 
                            initial_epoch=self.epoch, epochs=self.epoch + epochs, 
                            batch_size=batch_size, callbacks=[self.tb_callback], **kwargs)
        self.epoch += epochs
    def fit_mlp_classifier(self, epochs, batch_size=128, **kwargs):
        self.trainer_mlp_classifier.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), 
                            initial_epoch=self.epoch, epochs=self.epoch + epochs, 
                            batch_size=batch_size, callbacks=[self.tb_callback], **kwargs)
        self.epoch += epochs
    def fit_var_mlp_classifier(self, epochs, batch_size=128, **kwargs):
        self.trainer_var_mlp_classifier.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), 
                            initial_epoch=self.epoch, epochs=self.epoch + epochs, 
                            batch_size=batch_size, callbacks=[self.tb_callback], **kwargs)
        self.epoch += epochs