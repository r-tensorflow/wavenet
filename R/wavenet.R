#' Builds the wavenet model
#'
#' Builds the wavenet model as described in
#' \href{https://arxiv.org/abs/1609.03499}{van der Oord et al., \cite{WaveNet: A Generative Model for Raw Audio}}.
#'
#' @param input_shape input shape for the model (not including the axis dimension).
#'  Typically lenght 2 numeric vector. Used to build the `input_tensor` if no
#'  `input_tensor` is specified.
#' @param input_tensor Tensor to use as input for the model. Usually a 3d tensor.
#' @param residual_blocks if a single integer: the number of residual blocks in the
#'  model - the dilation rate of each block is calculated by `2^i`. if a vector,
#'  then it's used as the dilation rate of each residual block.
#' @param output_channels number of channels in the output.
#' @param output_activation activation function for the last layer. default: 'softmax'
#' @param initial_kernel_size kernel size of the first causal convolution.
#' @param initial_filters number of filters in the initial causal convolution.
#'
#' @inheritParams keras::layer_conv_1d
#'
#'
#' @importFrom zeallot %<-%
#' @export
wavenet <- function(filters = 16, kernel_size = 2, residual_blocks, input_shape = list(NULL, 1),
                    input_tensor = NULL, initial_kernel_size = 32, initial_filters = 32,
                    output_channels = 256, output_activation = "softmax") {

  if (is.null(input_shape) && is.null(input_tensor))
    stop("You must specify one of `input_shape` or `input_tensor`.", call. = FALSE)

  if (is.null(input_tensor)) {
    input <- keras::layer_input(shape = input_shape)
  } else {
    input <- input_tensor
  }

  # https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py#L46
  x <- keras::layer_conv_1d(
    object = input,
    filters = initial_filters,
    kernel_size = initial_kernel_size,
    padding = "causal"
  )

  skip_connections <- NULL

  if (length(residual_blocks) == 1) {
    dilation_rates <- 2^seq_len(residual_blocks)
  } else {
    dilation_rates <- residual_blocks
  }

  for (i in dilation_rates) {

    c(x, s) %<-% layer_wavenet_dilated_causal_convolution_1d(
      x,
      filters = filters,
      kernel_size = kernel_size,
      dilation_rate = i
    )

    skip_connections <- append(skip_connections, s)

  }

  output <- skip_connections %>%
    keras::layer_add() %>%
    keras::layer_activation(activation = "relu") %>%
    keras::layer_conv_1d(
      filters = initial_filters/2L, # reduces the number of filters
      kernel_size = 1,
      activation = "relu"
    ) %>%
    keras::layer_conv_1d(
      filters = output_channels,
      kernel_size = 1,
      activation = output_activation
    )

  keras::keras_model(input, output)
}
