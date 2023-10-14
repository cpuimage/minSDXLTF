import os

import tensorflow as tf

from stable_diffusion_xl.ckpt_loader import load_weights_from_file, UNET_KEY_MAPPING, CKPT_MAPPING


class LayerNormalization(tf.keras.layers.Layer):
    """Layer normalization layer (Ba et al., 2016).

    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.

    Given a tensor `inputs`, moments are calculated and normalization
    is performed across the axes specified in `axis`.

    Args:
      axis: Integer or List/Tuple. The axis or axes to normalize across.
        Typically this is the features axis/axes. The left-out axes are
        typically the batch axis/axes. This argument defaults to `-1`, the last
        dimension in the input.
      epsilon: Small float added to variance to avoid dividing by zero. Defaults
        to 1e-3
      center: If True, add offset of `beta` to normalized tensor. If False,
        `beta` is ignored. Defaults to True.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used.
        Defaults to True. When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling will be done by the next layer.
      beta_initializer: Initializer for the beta weight. Defaults to zeros.
      gamma_initializer: Initializer for the gamma weight. Defaults to ones.
      beta_regularizer: Optional regularizer for the beta weight. None by
        default.
      gamma_regularizer: Optional regularizer for the gamma weight. None by
        default.
      beta_constraint: Optional constraint for the beta weight. None by default.
      gamma_constraint: Optional constraint for the gamma weight. None by
        default.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape` (tuple of
      integers, does not include the samples axis) when using this layer as the
      first layer in a model.

    Output shape:
      Same shape as input.

    Reference:
      - [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).
    """

    def __init__(
            self,
            axis=-1,
            epsilon=1e-3,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.supports_masking = True
        # Indicates whether a faster fused implementation can be used. This will
        # be set to True or False in build()"
        self._fused = None

    def _fused_can_be_used(self, ndims):
        """Returns false if fused implementation cannot be used.

        Check if the axis is contiguous and can be collapsed into the last axis.
        The self.axis is assumed to have no duplicates.
        """
        axis = sorted(self.axis)
        can_use_fused = False

        if axis[-1] == ndims - 1 and axis[-1] - axis[0] == len(axis) - 1:
            can_use_fused = True
        # fused_batch_norm will silently raise epsilon to be at least 1.001e-5,
        # so we cannot used the fused version if epsilon is below that value.
        # Also, the variable dtype must be float32, as fused_batch_norm only
        # supports float32 variables.
        if self.epsilon < 1.001e-5 or self.dtype != "float32":
            can_use_fused = False
        return can_use_fused

    @staticmethod
    def validate_axis(axis, input_shape):
        """Validate an axis value and returns its standardized form.

        Args:
          axis: Value to validate. Can be an integer or a list/tuple of integers.
            Integers may be negative.
          input_shape: Reference input shape that the axis/axes refer to.

        Returns:
          Normalized form of `axis`, i.e. a list with all-positive values.
        """
        input_shape = tf.TensorShape(input_shape)
        rank = input_shape.rank
        if not rank:
            raise ValueError(
                f"Input has undefined rank. Received: input_shape={input_shape}")
        # Convert axis to list and resolve negatives
        if isinstance(axis, int):
            axis = [axis]
        else:
            axis = list(axis)
        for index, x in enumerate(axis):
            if x < 0:
                axis[index] = rank + x
        # Validate axes
        for x in axis:
            if x < 0 or x >= rank:
                raise ValueError(
                    "Invalid value for `axis` argument. "
                    "Expected 0 <= axis < inputs.rank (with "
                    f"inputs.rank={rank}). Received: axis={tuple(axis)}")
        if len(axis) != len(set(axis)):
            raise ValueError(f"Duplicate axis: {tuple(axis)}")
        return axis

    def build(self, input_shape):
        self.axis = self.validate_axis(self.axis, input_shape)
        input_shape = tf.TensorShape(input_shape)
        rank = input_shape.rank
        param_shape = [input_shape[dim] for dim in self.axis]
        if self.scale:
            self.gamma = self.add_weight(
                name="weight",
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                name="bias",
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.beta = None
        self._fused = self._fused_can_be_used(rank)
        self.built = True

    def call(self, inputs):
        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            inputs_lengths = inputs.nested_row_lengths()
            inputs = inputs.to_tensor()
        inputs = tf.cast(inputs, self.compute_dtype)
        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        # Broadcasting only necessary for norm when the axis is not just
        # the last dimension
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape.dims[dim].value

        def _broadcast(v):
            if (v is not None
                    and len(v.shape) != ndims
                    and self.axis != [ndims - 1]):
                return tf.reshape(v, broadcast_shape)
            return v

        if not self._fused:
            input_dtype = inputs.dtype
            if (input_dtype in ("float16", "bfloat16")
                    and self.dtype == "float32"):
                # If mixed precision is used, cast inputs to float32 so that
                # this is at least as numerically stable as the fused version.
                inputs = tf.cast(inputs, "float32")
            # Calculate the moments on the last axis (layer activations).
            mean, variance = tf.nn.moments(inputs, self.axis, keepdims=True)
            scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
            # Compute layer normalization using the batch_normalization
            # function.
            outputs = tf.nn.batch_normalization(
                inputs,
                mean,
                variance,
                offset=offset,
                scale=scale,
                variance_epsilon=self.epsilon,
            )
            outputs = tf.cast(outputs, input_dtype)
        else:
            # Collapse dims before self.axis, and dims in self.axis
            pre_dim, in_dim = (1, 1)
            axis = sorted(self.axis)
            tensor_shape = tf.shape(inputs)
            for dim in range(0, ndims):
                dim_tensor = tensor_shape[dim]
                if dim < axis[0]:
                    pre_dim = pre_dim * dim_tensor
                else:
                    assert dim in axis
                    in_dim = in_dim * dim_tensor
            squeezed_shape = [1, pre_dim, in_dim, 1]
            # This fused operation requires reshaped inputs to be NCHW.
            data_format = "NCHW"
            inputs = tf.reshape(inputs, squeezed_shape)
            # self.gamma and self.beta have the wrong shape for
            # fused_batch_norm, so we cannot pass them as the scale and offset
            # parameters. Therefore, we create two constant tensors in correct
            # shapes for fused_batch_norm and later construct a separate
            # calculation on the scale and offset.
            scale = tf.ones([pre_dim], dtype=self.dtype)
            offset = tf.zeros([pre_dim], dtype=self.dtype)
            # Compute layer normalization using the fused_batch_norm function.
            outputs, _, _ = tf.compat.v1.nn.fused_batch_norm(
                inputs,
                scale=scale,
                offset=offset,
                epsilon=self.epsilon,
                data_format=data_format,
            )
            outputs = tf.reshape(outputs, tensor_shape)
            scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
            if scale is not None:
                outputs = outputs * tf.cast(scale, outputs.dtype)
            if offset is not None:
                outputs = outputs + tf.cast(offset, outputs.dtype)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)
        if is_ragged:
            outputs = tf.RaggedTensor.from_tensor(outputs, inputs_lengths)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes nearly
    identical to Layer Normalization (see Layer Normalization docs for details).

    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is
    equal to number of channels), then this operation becomes identical to
    Instance Normalization.

    Args:
      groups: Integer, the number of groups for Group Normalization. Can be in
        the range [1, N] where N is the input dimension. The input dimension
        must be divisible by the number of groups. Defaults to 32.
      axis: Integer or List/Tuple. The axis or axes to normalize across.
        Typically this is the features axis/axes. The left-out axes are
        typically the batch axis/axes. This argument defaults to `-1`, the last
        dimension in the input.
      epsilon: Small float added to variance to avoid dividing by zero. Defaults
        to 1e-3
      center: If True, add offset of `beta` to normalized tensor. If False,
        `beta` is ignored. Defaults to True.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used.
        Defaults to True. When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling will be done by the next layer.
      beta_initializer: Initializer for the beta weight. Defaults to zeros.
      gamma_initializer: Initializer for the gamma weight. Defaults to ones.
      beta_regularizer: Optional regularizer for the beta weight. None by
        default.
      gamma_regularizer: Optional regularizer for the gamma weight. None by
        default.
      beta_constraint: Optional constraint for the beta weight. None by default.
      gamma_constraint: Optional constraint for the gamma weight. None by
        default.  Input shape: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis) when using this
        layer as the first layer in a model.  Output shape: Same shape as input.
    Reference: - [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)
    """

    def __init__(
            self,
            groups=32,
            axis=-1,
            epsilon=1e-3,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    @staticmethod
    def validate_axis(axis, input_shape):
        """Validate an axis value and returns its standardized form.

        Args:
          axis: Value to validate. Can be an integer or a list/tuple of integers.
            Integers may be negative.
          input_shape: Reference input shape that the axis/axes refer to.

        Returns:
          Normalized form of `axis`, i.e. a list with all-positive values.
        """
        input_shape = tf.TensorShape(input_shape)
        rank = input_shape.rank
        if not rank:
            raise ValueError(
                f"Input has undefined rank. Received: input_shape={input_shape}")
        # Convert axis to list and resolve negatives
        if isinstance(axis, int):
            axis = [axis]
        else:
            axis = list(axis)
        for index, x in enumerate(axis):
            if x < 0:
                axis[index] = rank + x
        # Validate axes
        for x in axis:
            if x < 0 or x >= rank:
                raise ValueError(
                    "Invalid value for `axis` argument. "
                    "Expected 0 <= axis < inputs.rank (with "
                    f"inputs.rank={rank}). Received: axis={tuple(axis)}")
        if len(axis) != len(set(axis)):
            raise ValueError(f"Duplicate axis: {tuple(axis)}")
        return axis

    def build(self, input_shape):
        self.validate_axis(self.axis, input_shape)
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                f"Axis {self.axis} of input tensor should have a defined "
                "dimension but the layer received an input with shape "
                f"{input_shape}.")

        if self.groups == -1:
            self.groups = dim

        if dim < self.groups:
            raise ValueError(
                f"Number of groups ({self.groups}) cannot be more than the "
                f"number of channels ({dim}).")

        if dim % self.groups != 0:
            raise ValueError(
                f"Number of groups ({self.groups}) must be a multiple "
                f"of the number of channels ({dim}).")

        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim})

        if self.scale:
            self.gamma = self.add_weight(
                shape=(dim,),
                name="weight",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint, )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                shape=(dim,),
                name="bias",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint, )
        else:
            self.beta = None

        super().build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        reshaped_inputs = self._reshape_into_groups(inputs)
        normalized_inputs = self._apply_normalization(
            reshaped_inputs, input_shape)
        return tf.reshape(normalized_inputs, input_shape)

    def _reshape_into_groups(self, inputs):
        input_shape = tf.shape(inputs)
        group_shape = [input_shape[index] for index in range(inputs.shape.rank)]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_reduction_axes = list(range(1, reshaped_inputs.shape.rank))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)
        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True)
        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon, )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * tf.keras.backend.int_shape(input_shape)[0]
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}


class Timesteps(tf.keras.layers.Layer):
    def __init__(self, num_channels: int = 320, trainable=False, name=None, **kwargs):
        super(Timesteps, self).__init__(name=name, trainable=trainable, **kwargs)
        self.num_channels = num_channels

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        half_dim = self.num_channels // 2
        exponent = -tf.math.log(10000.0) * tf.range(0, half_dim, dtype=tf.float32)
        exponent = exponent / (half_dim - 0.0)
        emb = tf.math.exp(exponent)
        emb = tf.cast(tf.expand_dims(inputs, axis=-1), tf.float32) * tf.expand_dims(emb, axis=0)
        sin_emb = tf.sin(emb)
        cos_emb = tf.cos(emb)
        emb = tf.concat([cos_emb, sin_emb], axis=-1)
        return emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_channels": self.num_channels,
        })
        return config


class SiLU(tf.keras.layers.Layer):
    def __init__(self, half: bool = True, trainable=False, name=None, **kwargs):
        super(SiLU, self).__init__(name=name, trainable=trainable, **kwargs)
        self.half = half

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.half:
            x = x * 0.5
        return x * (1.0 + tf.math.tanh(x))

    def get_config(self):
        config = super().get_config()
        config.update({
            "half": self.half,
        })
        return config


class Linear(tf.keras.layers.Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`). These are all attributes of
    `Dense`.

    Note: If the input to the layer has a rank greater than 2, then `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors).  The output in this case will have
    shape `(batch_size, d0, units)`.

    Besides, layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).
    When a popular kwarg `input_shape` is passed, then keras will create
    an input layer to insert before the current layer. This can be treated
    equivalent to explicitly defining an `InputLayer`.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "weight",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, *args, **kwargs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            # In case we encounter a RaggedTensor with a fixed last dimension
            # (last dimension not ragged), we can flatten the input and restore
            # the ragged dimensions at the end.
            if tf.compat.dimension_value(inputs.shape[-1]) is None:
                raise ValueError(
                    "Dense layer only supports RaggedTensors when the "
                    "innermost dimension is non-ragged. Received: "
                    f"inputs.shape={inputs.shape}."
                )
            original_inputs = inputs
            if inputs.flat_values.shape.rank > 1:
                inputs = inputs.flat_values
            else:
                # Innermost partition is encoded using uniform_row_length.
                # (This is unusual, but we can handle it.)
                if inputs.shape.rank == 2:
                    inputs = inputs.to_tensor()
                    is_ragged = False
                else:
                    for _ in range(original_inputs.ragged_rank - 1):
                        inputs = inputs.values
                    inputs = inputs.to_tensor()
                    original_inputs = tf.RaggedTensor.from_nested_row_splits(
                        inputs, original_inputs.nested_row_splits[:-1]
                    )

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul
            # operation for large sparse input tensors. The op will result in a
            # sparse gradient, as opposed to
            # sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, tf.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id
                # per row.
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding
                # lookup as a matrix multiply. We split our input matrix into
                # separate ids and weights tensors. The values of the ids tensor
                # should be the column indices of our input matrix and the
                # values of the weights tensor can continue to the actual matrix
                # weights.  The column arrangement of ids and weights will be
                # summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed
                # explanation of the inputs to both ops.
                ids = tf.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape,
                )
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(
                    self.kernel, ids, weights, combiner="sum"
                )
            else:
                outputs = tf.matmul(a=inputs, b=self.kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the input shape of a Dense layer "
                "should be defined. Found None. "
                f"Received: input_shape={input_shape}"
            )
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": tf.keras.regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": tf.keras.regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": tf.keras.constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
            }
        )
        return config


class GEGLU(tf.keras.layers.Layer):
    def __init__(self, out_features, trainable=True, name=None, **kwargs):
        super(GEGLU, self).__init__(name=name, trainable=trainable, **kwargs)
        self.proj = Linear(out_features * 2, use_bias=True, name="proj")

    def call(self, inputs, *args, **kwargs):
        x_proj = self.proj(inputs)
        x1, x2 = tf.split(x_proj, num_or_size_splits=2, axis=-1)
        return x1 * tf.nn.gelu(x2, approximate=True)


class Attention(tf.keras.layers.Layer):
    def __init__(self, query_dim, heads, dim_head, dropout=0.0, trainable=True, name=None, **kwargs):
        super(Attention, self).__init__(name=name, trainable=trainable, **kwargs)
        inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_size = dim_head
        self.drop = tf.keras.layers.Dropout(rate=dropout)
        self.to_q = Linear(inner_dim, use_bias=False, name="to_q")
        self.to_k = Linear(inner_dim, use_bias=False, name="to_k")
        self.to_v = Linear(inner_dim, use_bias=False, name="to_v")
        self.to_out = Linear(query_dim, use_bias=True, name="to_out.0")
        self.scale = dim_head ** -0.5

    def call(self, inputs, *args, **kwargs):
        hidden_states, encoder_hidden_states = inputs
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)
        batch_size = tf.shape(hidden_states)[0]
        q = tf.reshape(q, (batch_size, hidden_states.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (batch_size, -1, self.num_heads, self.head_size))
        v = tf.reshape(v, (batch_size, -1, self.num_heads, self.head_size))
        q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = tf.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        score = tf.einsum('bnqh,bnhk->bnqk', q, k) * self.scale
        weights = tf.keras.activations.softmax(score)  # (bs, num_heads, time, time)
        attn = tf.einsum('bnqk,bnkh->bnqh', weights, v)
        attn = tf.transpose(attn, (0, 2, 1, 3))  # (bs, time, num_heads, head_size)
        out = tf.reshape(attn, (-1, hidden_states.shape[1], self.num_heads * self.head_size))
        out = self.drop(out)
        return self.to_out(out)


class BasicTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_attention_heads, attention_head_dim, trainable=True, name=None, **kwargs):
        super(BasicTransformerBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        # 1. Self-Attn
        self.norm1 = LayerNormalization(epsilon=1e-05, center=True, scale=True, name="norm1")
        self.attn1 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, name="attn1")
        # 2. Cross-Attn
        self.norm2 = LayerNormalization(epsilon=1e-05, center=True, scale=True, name="norm2")
        self.attn2 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, name="attn2")
        # 3. Feed-forward
        self.norm3 = LayerNormalization(epsilon=1e-05, center=True, scale=True, name="norm3")
        self.ff0 = GEGLU(dim * 4, name="ff.net.0")
        self.ff2 = Linear(dim, name="ff.net.2")

    def call(self, inputs, *args, **kwargs):
        hidden_states, encoder_hidden_states = inputs
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1((norm_hidden_states, None))
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2((norm_hidden_states, encoder_hidden_states))
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff0(norm_hidden_states)
        ff_output = self.ff2(ff_output)
        hidden_states = ff_output + hidden_states
        return hidden_states


class DownSampler(tf.keras.layers.Layer):
    def __init__(self, out_channels, trainable=True, name=None, **kwargs):
        super(DownSampler, self).__init__(name=name, trainable=trainable, **kwargs)
        self.padding2d = tf.keras.layers.ZeroPadding2D(1)
        self.conv2d = tf.keras.layers.Conv2D(out_channels, 3, strides=2, name="conv")

    def call(self, inputs, *args, **kwargs):
        x = self.padding2d(inputs)
        return self.conv2d(x)


class UpSampler(tf.keras.layers.Layer):
    def __init__(self, out_channels, trainable=True, name=None, **kwargs):
        super(UpSampler, self).__init__(name=name, trainable=trainable, **kwargs)
        self.padding2d = tf.keras.layers.ZeroPadding2D(1)
        self.conv2d = tf.keras.layers.Conv2D(out_channels, 3, strides=1, name="conv")

    def call(self, inputs, *args, **kwargs):
        height, width = inputs.shape[1:3]
        x = tf.compat.v1.raw_ops.ResizeNearestNeighbor(images=inputs, size=(height + height, width + width),
                                                       align_corners=False, half_pixel_centers=False, name=None)
        x = self.padding2d(x)
        return self.conv2d(x)


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, groups=32, trainable=True, name=None, **kwargs):
        super(ResnetBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        self.norm1 = GroupNormalization(groups, epsilon=1e-05, center=True, scale=True, name="norm1")
        self.padding2d = tf.keras.layers.ZeroPadding2D(1)
        self.conv1 = tf.keras.layers.Conv2D(out_channels, 3, strides=1, name="conv1")
        self.time_emb_proj = Linear(out_channels, name="time_emb_proj")
        self.norm2 = GroupNormalization(groups, epsilon=1e-05, center=True, scale=True, name="norm2")
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, strides=1, name="conv2")
        self.nonlinearity = SiLU()
        self.conv_shortcut = None
        self.out_channels = out_channels

    def build(self, input_shape):
        axis = -1
        if input_shape[0][axis] != self.out_channels:
            self.conv_shortcut = tf.keras.layers.Conv2D(self.out_channels, kernel_size=1, strides=1,
                                                        name="conv_shortcut")

    def call(self, inputs, *args, **kwargs):
        hidden_states, time_emb = inputs
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(self.padding2d(x))
        emb = self.time_emb_proj(time_emb)
        emb = emb[:, None, None, :]
        x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(self.padding2d(x))
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32,
                 trainable=True, name=None, **kwargs):
        super(AttentionBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        inner_dim = num_attention_heads * attention_head_dim
        self.norm = GroupNormalization(norm_num_groups, epsilon=1e-06, center=True, scale=True,
                                       name="norm")
        self.proj_in = Linear(inner_dim, name="proj_in")
        self.transformer_blocks = [
            BasicTransformerBlock(inner_dim, num_attention_heads, attention_head_dim,
                                  name="transformer_blocks.{}".format(idx)) for idx in range(num_layers)]
        self.proj_out = Linear(in_channels, name="proj_out")

    def call(self, inputs, *args, **kwargs):
        hidden_states, text_emb = inputs
        batch = tf.shape(hidden_states)[0]
        height, width = hidden_states.shape[1:3]
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[-1]
        hidden_states = tf.reshape(hidden_states, (batch, height * width, inner_dim))
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                (hidden_states, text_emb))

        hidden_states = self.proj_out(hidden_states)
        hidden_states = tf.reshape(hidden_states, (batch, height, width, inner_dim))
        hidden_states = hidden_states + residual
        return hidden_states


class DiffusionXLModel(tf.keras.Model):
    @staticmethod
    def push_block(hidden_states, res_stack):
        res_stack.append(hidden_states)
        return res_stack

    @staticmethod
    def pop_block(hidden_states, res_stack):
        res_hidden_states = res_stack.pop()
        hidden_states = tf.concat([hidden_states, res_hidden_states], axis=-1)
        return hidden_states, res_stack

    def __init__(self, img_height=1024, img_width=1024, name=None, ckpt_path=None, lora_dict=None):
        sample = tf.keras.layers.Input((img_height // 8, img_width // 8, 4))
        timestep = tf.keras.layers.Input(())
        text_emb = tf.keras.layers.Input((77, 2048))
        text_embeds = tf.keras.layers.Input((1280,))
        time_ids = tf.keras.layers.Input((6,))
        # 1. time
        t_emb = Timesteps(320, name="time_proj")(timestep)
        t_emb = tf.reshape(t_emb, (-1, 320))
        t_emb = Linear(1280, name="time_embedding.linear_1")(tf.cast(t_emb, sample.dtype))
        t_emb = SiLU()(t_emb)
        t_emb = Linear(1280, name="time_embedding.linear_2")(t_emb)
        time_embeds = Timesteps(256, name="add_time_proj")(time_ids)
        time_embeds = tf.reshape(time_embeds, (-1, 1536))  # 6*256 = 1536
        add_embeds = tf.concat([text_embeds, time_embeds], axis=-1)
        add_embeds = tf.cast(add_embeds, sample.dtype)
        add_embeds = Linear(1280, name="add_embedding.linear_1")(add_embeds)
        add_embeds = SiLU()(add_embeds)
        add_embeds = Linear(1280, name="add_embedding.linear_2")(add_embeds)
        time_emb = SiLU()(t_emb + add_embeds)
        # 2. pre-process
        hidden_states = tf.keras.layers.Conv2D(320, kernel_size=3, strides=1, name="conv_in")(
            tf.keras.layers.ZeroPadding2D(1)(sample))
        res_stack = [hidden_states]
        # 3. blocks
        # DownBlock2D
        hidden_states = ResnetBlock(320, name="down_blocks.0.resnets.0")((hidden_states, time_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(320, name="down_blocks.0.resnets.1")((hidden_states, time_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = DownSampler(320, name="down_blocks.0.downsamplers.0")(hidden_states)
        res_stack = self.push_block(hidden_states, res_stack)
        # CrossAttnDownBlock2D
        hidden_states = ResnetBlock(640, name="down_blocks.1.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="down_blocks.1.attentions.0")((hidden_states, text_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(640, name="down_blocks.1.resnets.1")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="down_blocks.1.attentions.1")((hidden_states, text_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = DownSampler(640, name="down_blocks.1.downsamplers.0")(hidden_states)
        res_stack = self.push_block(hidden_states, res_stack)
        # CrossAttnDownBlock2D
        hidden_states = ResnetBlock(1280, name="down_blocks.2.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="down_blocks.2.attentions.0")((hidden_states, text_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(1280, name="down_blocks.2.resnets.1")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="down_blocks.2.attentions.1")((hidden_states, text_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        # UNetMidBlock2DCrossAttn
        hidden_states = ResnetBlock(1280, name="mid_block.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="mid_block.attentions.0")((hidden_states, text_emb))
        hidden_states = ResnetBlock(1280, name="mid_block.resnets.1")((hidden_states, time_emb))
        # CrossAttnUpBlock2D
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(1280, name="up_blocks.0.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="up_blocks.0.attentions.0")((hidden_states, text_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(1280, name="up_blocks.0.resnets.1")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="up_blocks.0.attentions.1")((hidden_states, text_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(1280, name="up_blocks.0.resnets.2")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="up_blocks.0.attentions.2")((hidden_states, text_emb))
        hidden_states = UpSampler(1280, name="up_blocks.0.upsamplers.0")(hidden_states)
        # CrossAttnUpBlock2D
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(640, name="up_blocks.1.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="up_blocks.1.attentions.0")((hidden_states, text_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(640, name="up_blocks.1.resnets.1")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="up_blocks.1.attentions.1")((hidden_states, text_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(640, name="up_blocks.1.resnets.2")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="up_blocks.1.attentions.2")((hidden_states, text_emb))
        hidden_states = UpSampler(640, name="up_blocks.1.upsamplers.0")(hidden_states)
        # UpBlock2D
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(320, name="up_blocks.2.resnets.0")((hidden_states, time_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(320, name="up_blocks.2.resnets.1")((hidden_states, time_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(320, name="up_blocks.2.resnets.2")((hidden_states, time_emb))
        hidden_states = GroupNormalization(32, epsilon=1e-05, center=True, scale=True,
                                           name="conv_norm_out")(
            hidden_states)
        hidden_states = SiLU()(hidden_states)
        output = tf.keras.layers.Conv2D(4, kernel_size=3, strides=1, name="conv_out")(
            tf.keras.layers.ZeroPadding2D(1)(hidden_states))
        super().__init__([sample, timestep, text_emb, time_ids, text_embeds], output, name=name)
        origin = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors"
        ckpt_mapping = CKPT_MAPPING["diffusion_model"]
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                load_weights_from_file(self, ckpt_path, ckpt_mapping=ckpt_mapping, key_mapping=UNET_KEY_MAPPING,
                                       lora_dict=lora_dict)
                return
            else:
                origin = ckpt_path
        model_weights_fpath = tf.keras.utils.get_file(origin=origin)
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping, key_mapping=UNET_KEY_MAPPING,
                                   lora_dict=lora_dict)
