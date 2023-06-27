import tensorflow as tf

class VAEC_Trainer(tf.keras.Model):
    def __init__(self, encoder, decoder, classifier_head, train_what, **kwargs):
        super().__init__(**kwargs)
        """"
        *Variational AutoEncoder + Classifier Trainer*
        Trains encoder, decoder or classifier_head.
        Parameter train_what controls training:
        - 'all': All 3 of encoder, decoder and classifier head are trained simultaneously.
        - 'autoencoder': Only encoder and decoder are being trained.
        - 'classifier_head': Only classifier_head is being trained.
        - 'mlp_classifier': Only encoder and classifier_head is being trained while randomization is switched off in Sampling layer.
        - 'var_mlp_classifier': Only encoder and classifier_head is being trained.
        """
        self.train_what = train_what
        self._encoder = encoder
        self._decoder = decoder
        self._classifier_head = classifier_head
        
        if self.train_what == 'all':
            self._encoder.trainable = True
            self._decoder.trainable = True
            self._classifier_head.trainable = True
        elif self.train_what == 'autoencoder':
            self._encoder.trainable = True
            self._decoder.trainable = True
            self._classifier_head.trainable = False
        elif self.train_what == 'classifier_head':
            self._encoder.trainable = False
            self._decoder.trainable = False
            self._classifier_head.trainable = True
        elif self.train_what == 'mlp_classifier':
            self._encoder.trainable = True
            self._decoder.trainable = False
            self._classifier_head.trainable = True
        elif self.train_what == 'var_mlp_classifier':
            self._encoder.trainable = True
            self._decoder.trainable = False
            self._classifier_head.trainable = True
        else:
            raise TypeError(
                "Parameter train_what must be 'all', 'autoencoder', 'classifier_head', 'mlp_classifier' or 'var_mlp_classifier'!")
        
        # Metrics to be tracked.
        self._total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self._recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self._kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self._pred_var_loss_tracker = tf.keras.metrics.Mean(name="pred_var_loss")
        self._accuracy_var = tf.keras.metrics.BinaryAccuracy(name="pred_var_acc")
        self._pred_loss_tracker = tf.keras.metrics.Mean(name="pred_loss")
        self._accuracy = tf.keras.metrics.BinaryAccuracy(name="pred_acc")

    @property
    def metrics(self):
        return [
            self._total_loss_tracker,
            self._recon_loss_tracker,
            self._kl_loss_tracker,
            self._pred_loss_tracker,
            self._accuracy,
            self._pred_var_loss_tracker,
            self._accuracy_var,
        ]
    
    def calc_loss(self, x, y):
        """
        Calculates losses for train_step() and test_step().
        """
        z_mean, z_log_var, z = self._encoder(x)
        recon = self._decoder(z)
        # if self.train_what == 'mlp_classifier':
        pred = self._classifier_head(z_mean)
        # else:
        pred_var = self._classifier_head(z)
        # Kullback Leibler Divergence loss.
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_mean(kl_loss, axis=1))
        # Reconstruction loss.
        recon_loss = tf.square(x-recon)
        recon_loss = tf.reduce_mean(tf.reduce_mean(recon_loss, axis=1))
        # Loss between input and autoencoder output.
        pred_loss = tf.keras.losses.binary_crossentropy(y, pred)
        pred_loss = tf.reduce_mean(pred_loss)
        
        pred_var_loss = tf.keras.losses.binary_crossentropy(y, pred_var)
        pred_var_loss = tf.reduce_mean(pred_var_loss)

        # Total loss is dependent on the training method.
        if self.train_what == 'all':
            total_loss = kl_loss + recon_loss + pred_var_loss
        elif self.train_what == 'autoencoder':
            total_loss = kl_loss + recon_loss
        elif self.train_what == 'classifier_head':
            total_loss = pred_var_loss
        elif self.train_what == 'mlp_classifier':
            total_loss = pred_loss
        elif self.train_what == 'var_mlp_classifier':
            total_loss = kl_loss + pred_var_loss
        else:
            raise TypeError("Something is not OK, train_what was checked in __init__.")

        return total_loss, recon_loss, kl_loss, pred_var_loss, pred_loss, pred_var, pred

    def train_step(self, data):
        """
        Custom train_step. Pure TF.
        """
        with tf.GradientTape() as tape:
            x = data[0]
            y = data[1]
            total_loss, recon_loss, kl_loss, pred_var_loss, pred_loss, pred_var, pred = \
                self.calc_loss(x, y)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Metrics to be tracked.
        self._total_loss_tracker.update_state(total_loss)
        self._recon_loss_tracker.update_state(recon_loss)
        self._kl_loss_tracker.update_state(kl_loss)
        self._pred_var_loss_tracker.update_state(pred_var_loss)
        self._accuracy_var.update_state(y, pred_var)
        self._pred_loss_tracker.update_state(pred_loss)
        self._accuracy.update_state(y, pred)

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "pred_var_loss": pred_var_loss,
            "accuracy_var": self._accuracy_var.result(),
            "pred_loss": pred_loss,
            "accuracy": self._accuracy.result(),
        }
    
    def test_step(self, data):
        """
        Custom test_step has been cerated just for making it possible that val_losses are shown in terminal too.
        """
        x = data[0]
        y = data[1]
        total_loss, recon_loss, kl_loss, pred_var_loss, pred_loss, pred_var, pred = \
                self.calc_loss(x, y)

        # Metrics to be tracked.
        self._total_loss_tracker.update_state(total_loss)
        self._recon_loss_tracker.update_state(recon_loss)
        self._kl_loss_tracker.update_state(kl_loss)
        self._pred_var_loss_tracker.update_state(pred_var_loss)
        self._accuracy_var.update_state(y, pred_var)
        self._pred_loss_tracker.update_state(pred_loss)
        self._accuracy.update_state(y, pred)

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "pred_var_loss": pred_var_loss,
            "accuracy_var": self._accuracy_var.result(),
            "pred_loss": pred_loss,
            "accuracy": self._accuracy.result(),
        }