#pragma once
#include "utils.h"
#include <thrust/extrema.h>

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
  return (x - 1) / y + 1;
}

struct MyStringColumnMetadata {
  int num_rows;
  int *offsets;
  int *lengths;
  bool onDevice;
  bool freed;
  MyStringColumnMetadata() = default;
  MyStringColumnMetadata(int num_rows, int *offsets, int *lengths,
                         bool onDevice) {
    this->num_rows = num_rows;
    this->offsets = offsets;
    this->lengths = lengths;
    this->onDevice = onDevice;
    this->freed = false;
  }

  MyStringColumnMetadata(const MyStringColumnMetadata &other) {
    this->num_rows = other.num_rows;
    this->offsets = other.offsets;
    this->lengths = other.lengths;
    this->onDevice = other.onDevice;
    this->freed = other.freed;
  }

  void free_() {
    if (!onDevice) {
      free(offsets);
      free(lengths);
    } else {
      cuda_err_chk(cudaFree(offsets));
      cuda_err_chk(cudaFree(lengths));
    }
    freed = true;
  }

  struct MyStringColumnMetadata copyToDevice() const {
    assert(!onDevice);
    int *offsets_d;
    int *lengths_d;
    cuda_err_chk(cudaMalloc(&offsets_d, num_rows * sizeof(int)));
    cuda_err_chk(cudaMalloc(&lengths_d, num_rows * sizeof(int)));
    cuda_err_chk(cudaMemcpy(offsets_d, offsets, num_rows * sizeof(int), cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(lengths_d, lengths, num_rows * sizeof(int), cudaMemcpyHostToDevice));

    MyStringColumnMetadata result(this->num_rows, offsets_d, lengths_d, true);
    return result;
  }

  struct MyStringColumnMetadata copyToHost() const {
    assert(onDevice);
    int *offsets_h;
    int *lengths_h;
    offsets_h = (int *)malloc(num_rows * sizeof(int));
    lengths_h = (int *)malloc(num_rows * sizeof(int));
    cuda_err_chk(cudaMemcpy(offsets_h, offsets, num_rows * sizeof(int),
                            cudaMemcpyDeviceToHost));
    cuda_err_chk(cudaMemcpy(lengths_h, lengths, num_rows * sizeof(int),
                            cudaMemcpyDeviceToHost));
    MyStringColumnMetadata result(this->num_rows, offsets_h, lengths_h, false);
    return result;
  }
  int *get_offsets_ptr() const {
    assert(!freed);
    return offsets;
  }
  int *get_lengths_ptr() const {
    assert(!freed);
    return lengths;
  }
};

template <bool transposed_templateflag>
__global__ void transpose_kernel_(char *out, char *in, int num_rows,
                                  int *lengths, int *offsets) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
    for (int j = 0; j < lengths[i]; j++) {
      if constexpr (transposed_templateflag) {
        out[j * num_rows + i] = in[offsets[i] + j];
      } else {
        out[offsets[i] + j] = in[j * num_rows + i];
      }
    }
  }
}
struct MyStringColumn {
  char *data;
  MyStringColumnMetadata metadata;
  int data_size;
  bool onDevice;
  bool freed;
  bool transposed;
  MyStringColumn() = default;
  MyStringColumn(char *data, const MyStringColumnMetadata &metadata,
                 int data_size, bool onDevice, bool transposed) {
    this->data = data;
    this->metadata = metadata;
    this->onDevice = onDevice;
    this->freed = false;
    this->data_size = data_size;
    this->transposed = transposed;
  }
  static void FreeStructOnDevice_(MyStringColumn *column) {
    struct MyStringColumn column_h;
    cuda_err_chk(cudaMemcpy(&column_h, column, sizeof(struct MyStringColumn),
                            cudaMemcpyDeviceToHost));

    column_h.free_();
  }
  void free_() {
    if (!onDevice) {
      free(data);
    } else {
      cuda_err_chk(cudaFree(data));
    }
    metadata.free_();
    freed = true;
  }
  struct MyStringColumn copyToDevice() const {
    assert(!onDevice);
    char *data_d;
    cuda_err_chk(cudaMalloc(&data_d, data_size * sizeof(char)));
    cuda_err_chk(cudaMemcpy(data_d, data, data_size * sizeof(char),
                            cudaMemcpyHostToDevice));
    MyStringColumn result(data_d, metadata.copyToDevice(), data_size, true,
                          transposed);
    return result;
  }
  static struct MyStringColumn *
  CopyToDevice(const struct MyStringColumn &column, bool exec_transpose) {
    struct MyStringColumn result = column.copyToDevice();
    if (exec_transpose) {
      result.transpose();
    }
    struct MyStringColumn *result_ptr;
    cuda_err_chk(cudaMalloc(&result_ptr, sizeof(struct MyStringColumn)));
    cuda_err_chk(cudaMemcpy(result_ptr, &result, sizeof(struct MyStringColumn),
                            cudaMemcpyHostToDevice));
    return result_ptr;
  }
  struct MyStringColumn copyToHost() const {
    assert(onDevice);
    char *data_h;
    data_h = (char *)malloc(data_size * sizeof(char));
    cuda_err_chk(cudaMemcpy(data_h, data, data_size * sizeof(char),
                            cudaMemcpyDeviceToHost));
    MyStringColumn result(data_h, metadata.copyToHost(), data_size, false,
                          transposed);
    return result;
  }
  static struct MyStringColumn CopyToHost(const struct MyStringColumn *column) {

    struct MyStringColumn column_h;
    cuda_err_chk(cudaMemcpy(&column_h, column, sizeof(struct MyStringColumn),
                            cudaMemcpyDeviceToHost));
    struct MyStringColumn result = column_h.copyToHost();
    return result;
  }

  char *get_data_ptr() const {
    assert(!freed);
    return data;
  }

  void transpose() {
    if (!onDevice) {
      if (!transposed) {
        int max_string_length = *(thrust::max_element(
            metadata.get_lengths_ptr(),
            metadata.get_lengths_ptr() + metadata.num_rows));
        data_size = (max_string_length * metadata.num_rows) * sizeof(char);
        char *data_transposed = (char *)malloc(
            (max_string_length * metadata.num_rows) * sizeof(char));
        for (int i = 0; i < metadata.num_rows; i++) {
          for (int j = 0; j < metadata.get_lengths_ptr()[i]; j++) {
            data_transposed[j * metadata.num_rows + i] =
                data[metadata.offsets[i] +
                     j]; // TODO: implement chunk-transpose
          }
        }
        free(data);
        transposed = true;
        data = data_transposed;
      } else {
        // int max_string_length =
        // *(thrust::max_element(metadata.get_lengths_ptr(),
        // metadata.get_lengths_ptr() + metadata.num_rows));

        data_size = (metadata.offsets[metadata.num_rows - 1] +
                     metadata.lengths[metadata.num_rows - 1]);
        char *data_untransposed =
            (char *)malloc((metadata.offsets[metadata.num_rows - 1] +
                            metadata.lengths[metadata.num_rows - 1]) *
                           sizeof(char));
        for (int i = 0; i < metadata.num_rows; i++) {
          for (int j = 0; j < metadata.get_lengths_ptr()[i]; j++) {
            data_untransposed[metadata.offsets[i] + j] =
                data[j * metadata.num_rows +
                     i]; // TODO: implement chunk-transpose
          }
        }
        free(data);
        transposed = false;
        data = data_untransposed;
      }
    } else {
      if (!transposed) {
        int max_string_length;
        cuda_err_chk(
            cudaMemcpy(&max_string_length,
                       (thrust::max_element(metadata.get_lengths_ptr(),
                                            metadata.get_lengths_ptr() +
                                                metadata.num_rows)),
                       sizeof(int), cudaMemcpyDeviceToHost));

        data_size = (max_string_length * metadata.num_rows) * sizeof(char);
        char *data_transposed;
        cuda_err_chk(cudaMalloc(&data_transposed, data_size));
        transpose_kernel_<true><<<256, ceil_div(metadata.num_rows, 256)>>>(
            data_transposed, data, metadata.num_rows, metadata.lengths,
            metadata.offsets); // TODO: implement chunk-transpose (basically
                               // copy paste the assignment statement from the
                               // previous CPU logic once you finished it)

        cuda_err_chk(cudaFree(data));
        transposed = true;
        data = data_transposed;

      } else {
        int last_row_offset;
        int last_row_length;
        cuda_err_chk(cudaMemcpy(&last_row_offset,
                                &metadata.offsets[metadata.num_rows - 1],
                                sizeof(int), cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemcpy(&last_row_length,
                                &metadata.lengths[metadata.num_rows - 1],
                                sizeof(int), cudaMemcpyDeviceToHost));
        data_size = (last_row_offset + last_row_length) * sizeof(char);
        char *data_untransposed;
        cuda_err_chk(cudaMalloc(&data_untransposed, data_size));

        transpose_kernel_<false><<<256, ceil_div(metadata.num_rows, 256)>>>(
            data_untransposed, data, metadata.num_rows, metadata.lengths,
            metadata.offsets); // TODO: implement chunk-transpose (basically
                               // copy paste the assignment statement from the
                               // previous CPU logic once you finished it)
        cuda_err_chk(cudaFree(data));
        transposed = false;
        data = data_untransposed;
      }
    }
  }

  static void TransposeStructOnDevice(MyStringColumn *column) {
    struct MyStringColumn column_h;
    cuda_err_chk(cudaMemcpy(&column_h, column, sizeof(struct MyStringColumn),
                            cudaMemcpyDeviceToHost));
    column_h.transpose();
    cuda_err_chk(cudaMemcpy(column, &column_h, sizeof(struct MyStringColumn),
                            cudaMemcpyHostToDevice));
  }
  template <bool transposed_templateflag>
  __inline__ __host__ __device__ char get_char_(int idx_row, int idx_char) {
    if constexpr (!transposed_templateflag) {
      return data[metadata.offsets[idx_row] + idx_char];
    } else {
      return data[idx_char * metadata.num_rows + idx_row];
    }
  }

  // writing in this way won't inline get_char_ nvcc version 11.5
  // http://websites.umich.edu/~eecs381/handouts/Pointers_to_memberfuncs.pdf
  __inline__ __host__ __device__ char get_char_experimental(int idx_row,
                                                            int idx_char) {
    char (MyStringColumn::*get_char_ptr)(int, int);
    if (!transposed) {
      // return get_char_<false>(idx_row, idx_char);
      get_char_ptr = &MyStringColumn::get_char_<false>;
    } else {
      get_char_ptr = &MyStringColumn::get_char_<true>;
      // return get_char_<true>(idx_row, idx_char);
    }
    return (this->*(get_char_ptr))(idx_row, idx_char);
  }

  // writing in this way inline get_char_ nvcc version 11.5
  __inline__ __host__ __device__ char get_char(int idx_row, int idx_char) {
    if (!transposed) {
      return get_char_<false>(idx_row, idx_char);
    } else {
      return get_char_<true>(idx_row, idx_char);
    }
  }
};


