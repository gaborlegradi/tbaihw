import tensorflow as tf

class Sampling(tf.keras.layers.Layer):
    """
    Sampling layer for variational encoder.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_encoder(global_input_dim, encoder_layers, latent_dim, activation='relu', summary=False):
    """
    Creates encoder part as tf.keras.Model.
    Input: Of global_input_dim size.
    Output: z_mean, z_log_var, z of latent_dim size.
    """
    encoder_inputs = tf.keras.Input(shape=(global_input_dim,))
    x = encoder_inputs
    for num_neurons in encoder_layers:
        x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    if summary: encoder.summary()
    
    return encoder

def create_decoder(global_input_dim, decoder_layers, latent_dim, activation='relu', summary=False):
    """
    Creates decoder part as tf.keras.Model.
    Input: Of latent_dim size.
    Output: Of global_input_dim size.
    """
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = latent_inputs
    for num_neurons in decoder_layers:
        x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
    decoder_outputs = tf.keras.layers.Dense(global_input_dim, activation="linear")(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    if summary: decoder.summary()

    return decoder

def create_classifier_head(latent_dim, classifier_layers, activation='relu', summary=False):
    """
    Creates classifier_head part as tf.keras.Model.
    Input: Of latent_dim size.
    Output: Single neuron with sigmoid activation.
    """
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    z = latent_inputs
    for num_neurons in classifier_layers:
        z = tf.keras.layers.Dense(num_neurons, activation=activation)(z)
    classifier_outputs = tf.keras.layers.Dense(1, activation="sigmoid")(z)
    classifier_head = tf.keras.Model(latent_inputs, classifier_outputs, name="classifier_head")

    if summary: classifier_head.summary()

    return classifier_head