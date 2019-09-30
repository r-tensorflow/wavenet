WaveNetDilatedCausalConvolution1D <- R6::R6Class(
  "WaveNetDilatedCausalConvolution",
  inherit = keras::KerasLayer,
  public = list(

    filters = NULL,
    kernel_size = NULL,
    dilation_rate = NULL,

    initialize = function(filters, kernel_size, dilation_rate) {

      self$filters <- filters
      self$kernel_size <- kernel_size
      self$dilation_rate <- dilation_rate

    },

    conv_sigmoid = NULL,
    conv_tanh = NULL,
    conv_1x1 = NULL,
    conv_1x1_filters = NULL,


    build = function(input_shape) {

      # https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/model.py#L245

      self$conv_sigmoid <- keras::layer_conv_1d(
        filters = self$filters,
        kernel_size = self$kernel_size,
        dilation_rate = self$dilation_rate,
        activation = "sigmoid",
        padding = "causal"
      )

      self$conv_tanh <- keras::layer_conv_1d(
        filters = self$filters,
        kernel_size = self$kernel_size,
        dilation_rate = self$dilation_rate,
        activation = "tanh",
        padding = "causal"
      )

      self$conv_1x1_filters <- input_shape[[3]]

      self$conv_1x1 <- keras::layer_conv_1d(
        filters = self$conv_1x1_filters,
        kernel_size = 1
      )

    },

    call = function(x, mask = NULL) {

      out <- keras::layer_multiply(
        list(
          self$conv_sigmoid(x),
          self$conv_tanh(x)
        )
      )

      out <- self$conv_1x1(out)

      residual <- keras::layer_add(
        list(
          x,
          out
        )
      )

      list(
        residual,
        out
      )
    },

    compute_output_shape = function(input_shape) {
      list(
        input_shape,
        input_shape
      )
    }

  )
)

#' Wavenet residual connections
#'
#' Residual connection as described in section 2.3 of
#' \href{https://arxiv.org/abs/1609.03499}{van der Oord et al., \cite{WaveNet: A Generative Model for Raw Audio}}.
#'
#' @inheritParams keras::layer_conv_1d
#'
#' @export
layer_wavenet_dilated_causal_convolution_1d <- function(object, filters, kernel_size,
                                                        dilation_rate, name = NULL,
                                                        trainable = TRUE) {
  keras::create_layer(WaveNetDilatedCausalConvolution1D, object, list(
    filters = filters,
    kernel_size = kernel_size,
    dilation_rate = dilation_rate,
    name = name,
    trainable = trainable
  ))
}


