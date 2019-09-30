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
#' @param include_top wether to include the 256 softmax output layer.
#'
#' @inheritParams keras::layer_conv_1d
#'
#'
#' @importFrom zeallot %<-%
#' @export
wavenet <- function(filters, kernel_size, residual_blocks, input_shape = NULL,
                    input_tensor = NULL, include_top = TRUE) {

  if (is.null(input_shape) && is.null(input_tensor))
    stop("You must specify one of `input_shape` or `input_tensor`.", call. = FALSE)

  if (is.null(input_tensor)) {
    input <- keras::layer_input(shape = input_shape)
  } else {
    input <- input_tensor
  }

  x <- keras::layer_conv_1d(
    object = input,
    filters = filters,
    kernel_size = kernel_size,
    dilation_rate = 1,
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
    keras::layer_conv_1d(filters = 1, kernel_size = 1, activation = "relu") %>%
    keras::layer_conv_1d(filters = 1, kernel_size = 1) %>%
    keras::layer_flatten()

  if (include_top) {
    output <- output %>%
      keras::layer_dense(units = 256, activation = "softmax")
  }

  keras::keras_model(input, output)
}
