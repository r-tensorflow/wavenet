library(tfdatasets)
library(purrr)

# purrr::walk(
#   glue::glue("sox data-raw/{1:50}.wav -r 16000 data-raw2/{1:50}.wav"),
#   system
# )

wavs <- fs::dir_ls("data-raw/") %>%
  tensor_slices_dataset() %>%
  dataset_repeat(20) %>%
  dataset_shuffle(buffer_size = 64) %>%
  dataset_map(~.x %>%  tf$io$read_file() %>% tf$audio$decode_wav()) %>%
  dataset_map(~random_crop(.x, seconds = 5)) %>%
  dataset_map(~list(
    .x,
    .x %>% tf$reduce_mean(axis = 1L) %>% mu_law_and_one_hot_encode()
    )
  ) %>%
  dataset_batch(32)


it <- reticulate::as_iterator(wavs)
x <- reticulate::iter_next(it)

model <- wavenet(residual_blocks = 2^rep(1:8, 3), input_shape = list(NULL, 2))

system.time({
  out <- model(x[[1]])
})

loss <- function(y_true, y_pred) {
  keras::loss_categorical_crossentropy(
    y_true[, `2:`, ],
    y_pred[, `:-2`, ]
  )
}

model %>%
  keras::compile(
    loss = loss,
    optimizer = "adam"
  )

model %>%
  fit(wavs, epochs = 10)


