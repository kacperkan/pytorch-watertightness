#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <vector>
#include <iostream>

#include "src/watertightness.cuh"
#include "src/utils.hpp"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor watertightness(
    at::Tensor ray_origins,
    at::Tensor ray_directions,
    at::Tensor triangles
) {
    int64_t num_points = ray_origins.size(0);    
    int64_t num_triangles = triangles.size(0);

    at::Tensor test = torch::empty(
        {num_points}, 
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(ray_origins.device())
    );
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_directions);
    CHECK_INPUT(triangles);
    
    watertightness(
        ray_origins.data<float>(),
        ray_directions.data<float>(),
        triangles.data<float>(),
        test.data<float>(),
        num_points,
        num_triangles,
        at::cuda::getCurrentCUDAStream()
    );
    return test;
}
