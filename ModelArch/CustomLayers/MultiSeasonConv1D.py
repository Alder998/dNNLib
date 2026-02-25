import tensorflow as tf

class MultiSeasonalGatedConv1D(tf.keras.layers.Layer):
    def __init__(self, cycles, units_per_cycle, use_layer_norm, mix_units=None, baseline_kernel=None, **kwargs):
        super().__init__(**kwargs)

        # List or scalars are accepted
        if isinstance(cycles, int):
            cycles = [cycles]
        if isinstance(units_per_cycle, int):
            units_per_cycle = [units_per_cycle] * len(cycles)

        # Implement a way to get a precise error when the number of cycles is not the same number as units
        if len(cycles) != len(units_per_cycle):
            raise ValueError(f"cycles ({len(cycles)}) and units_per_cycle ({len(units_per_cycle)}) must match")

        self.cycles = cycles
        self.units_per_cycle = units_per_cycle
        self.mix_units = mix_units or sum(units_per_cycle)
        self.use_layer_norm = use_layer_norm

        # Baseline Convolution
        # baseline kernel = longest cycle if not specified
        self.baseline_kernel = baseline_kernel or max(cycles)

        # baseline conv (1 filtro = livello)
        self.baseline_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.baseline_kernel,
            padding="same"
        )

        self.convs = []
        self.gates = []

        for c, u in zip(self.cycles, self.units_per_cycle):
            # Convolutional Part
            self.convs.append(
                tf.keras.layers.Conv1D(filters=u, kernel_size=c, padding="same"))
            # Apply gates to convolutions
            self.gates.append(
                tf.keras.layers.Conv1D(filters=u,kernel_size=c, padding="same", activation="sigmoid"))

        # Add the 1x1 Mixing Conv1D to better include the seasonality
        self.mix_conv = tf.keras.layers.Conv1D(
            filters=self.mix_units,
            kernel_size=1,
            padding="same",
            activation="relu"
        )
        # add a layer norm, if required
        if self.use_layer_norm:
            self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        else:
            self.layer_norm = None

    def call(self, inputs):
        # Baseline used as Mobile support
        baseline = self.baseline_conv(inputs)  # (B,T,1)

        # detrended series
        x = inputs - baseline

        outputs = []
        for conv, gate in zip(self.convs, self.gates):
            s = conv(x)
            g = gate(x)
            outputs.append(s * g)

        x = tf.concat(outputs, axis=-1)
        x = self.mix_conv(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # restore baseline level
        x = x + baseline

        return x

    def compute_output_shape(self, input_shape):
        batch, time, _ = input_shape
        total_units = sum(self.units_per_cycle)
        return (batch, time, total_units)

    def get_config(self):
        config = super().get_config()
        config.update({
            "cycles": self.cycles,
            "units_per_cycle": self.units_per_cycle})
        return config