#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "lightx2v_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(lightx2v_kernel, m) {

  m.def(
      "cutlass_scaled_fp4_mm_sm120(Tensor! out, Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, Tensor "
      "alpha, Tensor? bias) -> ()");
  m.impl("cutlass_scaled_fp4_mm_sm120", torch::kCUDA, &cutlass_scaled_fp4_mm_sm120);

  m.def(
      "scaled_fp4_quant_sm120(Tensor! output, Tensor! input,"
      "                 Tensor! output_scale, Tensor! input_scale) -> ()");
  m.impl("scaled_fp4_quant_sm120", torch::kCUDA, &scaled_fp4_quant_sm120);

  m.def(
      "cutlass_scaled_mxfp6_mxfp8_mm_sm120(Tensor! out, Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, Tensor "
      "alpha, Tensor? bias) -> ()");
  m.impl("cutlass_scaled_mxfp6_mxfp8_mm_sm120", torch::kCUDA, &cutlass_scaled_mxfp6_mxfp8_mm_sm120);

}

REGISTER_EXTENSION(common_ops)
