#' Takes a random crop of an audio Tensor
#'
#' @param x An audio file decode by `tensorflow::tf$audio$decode_wav`.
#' @param seconds The number of seconds of the random crop.
#'
#' @importFrom tensorflow tf
#'
#' @export
random_crop <- function(x, seconds) {

  size <- as.integer(seconds)*x$sample_rate
  audio_shape <- tf$shape(x$audio)


  # randomly select an starting point
  start <- tf$random$uniform(
    shape = list(1L),
    minval = 1L,
    maxval = audio_shape[1] - size,
    dtype = tf$int32
  )

  # slice the audio
  audio <- tf$slice(
    input_ = x$audio,
    begin = tf$concat(
      list(start, tf$constant(0L, shape = list(1L))),
      axis = 0L
    ),
    size = tf$concat(
      list(
        tf$expand_dims(size, 0L),
        tf$constant(-1L, shape = list(1L))
      ),
      axis = 0L
    )
  )

  audio
}

#' Mu Law encodes the audio
#'
#' See section 2.2 of \href{https://arxiv.org/abs/1609.03499}{van der Oord et al., \cite{WaveNet: A Generative Model for Raw Audio}}.
#' for more information.
#'
#' @param x Tensor representing an audio.
#' @param mu quantization constant.
#'
#' @export
mu_law <- function(x, mu = 255) {
  tf$sign(x) * (tf$math$log(1 + mu * tf$abs(x))/tf$math$log(1 + mu))
}

#' Makes the transformation proposed in WaveNet and One Hot encodes the output
#'
#' See section 2.2 of \href{https://arxiv.org/abs/1609.03499}{van der Oord et al., \cite{WaveNet: A Generative Model for Raw Audio}}.
#' for more information.
#'
#' @inheritParams mu_law
#' @export
mu_law_and_one_hot_encode <- function(x, mu) {
  tf$cast((mu_law(x) + 1)/2*255, tf$uint8) %>%
    tf$one_hot(depth = 256L, axis = 1L)
}


