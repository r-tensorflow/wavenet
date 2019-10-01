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
