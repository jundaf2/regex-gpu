#include "MyStringColumn.cuh"
#include "StringColumnLoader.cuh"
#include <cstdlib>
#include <ctime>

template <typename T>
__global__ void checkEquivalent(T *arrA, T *arrB, int *out, size_t len) {
  if (threadIdx.x + blockIdx.x * blockDim.x == 0)
    out[0] = 0;
  __syncthreads();
  for (int stride = 0; stride < len; stride += gridDim.x * blockDim.x) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x + stride;
    if (idx < len) {
      if (arrA[idx] != arrB[idx]) {
        // printf("wrong %d\n", idx);
        atomicAdd(out, 1);
      }
    }
  }
}

template <typename T>
void checkCPUEquivalent(T *arrA, T *arrB, int *out, size_t len) {
  for (int i = 0; i < len; i++) {
    if (arrA[i] != arrB[i]) {
      out[0] += 1;
    }
  }
}

void generate_random_lengths(int *lengths, int num_rows, int max_length) {
  for (int i = 0; i < num_rows; i++) {
    lengths[i] = rand() % max_length;
  }
}

void generate_random_data(char *data, int *lengths, int *offsets,
                          int num_rows) {
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < lengths[i]; j++) {
      data[offsets[i] + j] = rand() % 128;
    }
  }
}

// __global__ void access_data(MyStringColumn *stringcolumn_ptr) {
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   for (int idx = tid; idx < stringcolumn_ptr->metadata.num_rows;
//        idx += blockDim.x * gridDim.x) {
//     stringcolumn_ptr->data[idx];
//     // if (stringcolumn_ptr->metadata. == '\0'){
//     //    printf("%d\n", idx);
//     //}
//   }
// }

int CPU_sum_up_first_char(MyStringColumn &stringcolumn) {
  int result = 0;
  assert(!stringcolumn.onDevice);
  for (int idx = 0; idx < stringcolumn.metadata.num_rows; idx++) {
    if (stringcolumn.metadata.lengths[idx] > 0) {
      result += stringcolumn.get_char(idx, 0);
    }
  }
  return result;
}

int main() {
  int num_rows = 100;
  int *offsets;
  int *lengths;
  int num_discrepency;
  lengths = (int *)malloc(num_rows * sizeof(int));
  offsets = (int *)malloc(num_rows * sizeof(int));
  generate_random_lengths(lengths, num_rows, 200);
  get_offsets_from_lengths(offsets, lengths, num_rows);
  char *string_data = (char *)malloc(
      (lengths[num_rows - 1] + offsets[num_rows - 1]) * sizeof(char));
  generate_random_data(string_data, lengths, offsets, num_rows);
  char *data_ref = (char *)malloc(
      (lengths[num_rows - 1] + offsets[num_rows - 1]) * sizeof(char));
  memcpy(data_ref, string_data,
         (lengths[num_rows - 1] + offsets[num_rows - 1]) * sizeof(char));

  MyStringColumnMetadata metadata(num_rows, offsets, lengths, false);
  MyStringColumn column(string_data, metadata,
                        lengths[num_rows - 1] + offsets[num_rows - 1], false,
                        false);
  assert(column.metadata.num_rows == metadata.num_rows);
  assert(column.metadata.onDevice == metadata.onDevice);
  assert(column.metadata.offsets == metadata.offsets);
  assert(column.metadata.lengths == metadata.lengths);
  assert(column.metadata.freed == metadata.freed);
  std::cout << column.data_size << std::endl;
  column.transpose();
  MyStringColumn *column_d = MyStringColumn::CopyToDevice(column, false);
  MyStringColumn column_d_to_h = MyStringColumn::CopyToHost(column_d);
  assert(column_d_to_h.data_size == column.data_size);
  checkCPUEquivalent(column.get_data_ptr(), column_d_to_h.get_data_ptr(),
                     &num_discrepency, column.data_size);
  checkCPUEquivalent(column.metadata.offsets, column_d_to_h.metadata.offsets,
                     &num_discrepency, num_rows);
  checkCPUEquivalent(column.metadata.lengths, column_d_to_h.metadata.lengths,
                     &num_discrepency, num_rows);
  column_d_to_h.transpose();
  checkCPUEquivalent(data_ref, column_d_to_h.get_data_ptr(), &num_discrepency,
                     lengths[num_rows - 1] + offsets[num_rows - 1]);
  std::cout << column.data_size << "  " << column_d_to_h.data_size << "   "
            << lengths[num_rows - 1] + offsets[num_rows - 1] << std::endl;
  assert(column_d_to_h.data_size ==
         lengths[num_rows - 1] + offsets[num_rows - 1]);

  MyStringColumn *column_d_transposed =
      MyStringColumn::CopyToDevice(column, true);
  int *first_char_sum_up_d;
  int first_char_sum_up_h = 0, first_char_sum_up_cpu = 0;
  cuda_err_chk(cudaMalloc((void **)&first_char_sum_up_d, sizeof(int)));
  cuda_err_chk(cudaMemset(first_char_sum_up_d, 0, sizeof(int)));
  sum_up_first_char<<<256, ceil_div(column.metadata.num_rows, 256)>>>(
      column_d_transposed, first_char_sum_up_d);
  cuda_err_chk(cudaMemcpy(&first_char_sum_up_h, first_char_sum_up_d,
                          sizeof(int), cudaMemcpyDeviceToHost));
  MyStringColumn column_d_transposed_to_h =
      MyStringColumn::CopyToHost(column_d_transposed);
  first_char_sum_up_cpu = CPU_sum_up_first_char(column_d_transposed_to_h);
  assert(column_d_transposed_to_h.data_size ==
         lengths[num_rows - 1] + offsets[num_rows - 1]);
  checkCPUEquivalent(data_ref, column_d_transposed_to_h.get_data_ptr(),
                     &num_discrepency,
                     lengths[num_rows - 1] + offsets[num_rows - 1]);
  std::cout << "first char sum up " << first_char_sum_up_h << " "
            << first_char_sum_up_cpu << std::endl;
  assert(first_char_sum_up_h == first_char_sum_up_cpu);
  std::cout << "length_raw_data: "
            << lengths[num_rows - 1] + offsets[num_rows - 1] << std::endl;
  std::cout << "num_discrepency: " << num_discrepency << std::endl;
  assert(num_discrepency == 0);

  free(data_ref);
  column.free_();
  MyStringColumn::FreeStructOnDevice_(column_d);
}