#include "src/utils.hpp"

#define min(a, b) a < b ? a : b

__device__ void cross(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    float* __restrict__ c
) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ void sub(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    float* __restrict__ c
) {
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}

__device__ float dot(const float* __restrict__ a, const float* __restrict__ b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ float signed_volume(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    const float* __restrict__ c, 
    const float* __restrict__ d
) {
    float diff_b_a[3];
    float diff_c_a[3];
    float diff_d_a[3];

    float cross_diff_b_a_diff_c_a[3];

    sub(b, a, diff_b_a);
    sub(c, a, diff_c_a);
    sub(d, a, diff_d_a);

    cross(diff_b_a, diff_c_a, cross_diff_b_a_diff_c_a);

    return 1.0 / 6.0 * dot(cross_diff_b_a_diff_c_a, diff_d_a);
}

__device__ bool same_sign(float value_1, float value_2) {
    return (int)(value_1 < 0) == (int)(value_2 < 0);
}

__device__ bool same_sign3(float value_1, float value_2, float value_3) {
    return (value_1 < 0) == (value_2 < 0) && (value_1 < 0) == (value_3 < 0) && (value_2 < 0) == (value_3 < 0);
}

__device__ float signed_area(
    const float * __restrict__ x1, 
    const float * __restrict__ x2, 
    const float * __restrict__ a, 
    const float * __restrict__ w
) {
    float diff_x1_a[3];
    sub(x1, a, diff_x1_a);

    float diff_x2_a[3];
    sub(x2, a, diff_x2_a);

    float cross_diffs[3];
    cross(diff_x1_a, diff_x2_a, cross_diffs);

    return 0.5 * dot(cross_diffs, w);
}



__global__ void watertightness_kernel(
    const float* __restrict__  ray_origins, 
    const float* __restrict__ ray_directions, 
    const float* __restrict__ triangles, 
    float* __restrict__ passed_test,
    int n_rays, 
    int n_triangles
) {
    for (
        int i = blockDim.x * blockIdx.x + threadIdx.x; 
        i < n_rays; 
        i += gridDim.x
    ) {
        int num_intersections = 0;
        for (int triangle_i = 0 ; triangle_i < n_triangles; ++triangle_i) {
            const float *current_triangle = &triangles[triangle_i * 3 * 3];
            const float *p1 = &current_triangle[0];
            const float *p2 = &current_triangle[3];
            const float *p3 = &current_triangle[6];
            
            float a1 = signed_area(p1, p2, &ray_origins[i * 3], &ray_directions[i * 3]);
            float a2 = signed_area(p2, p3, &ray_origins[i * 3], &ray_directions[i * 3]);
            float a3 = signed_area(p3, p1, &ray_origins[i * 3], &ray_directions[i * 3]);

            if (same_sign3(a1, a2, a3)) {
                num_intersections++;
            }
        }
        passed_test[i] = (float)(num_intersections % 2 == 0);
    }
}


void watertightness(
    const float* ray_origins, 
    const float* ray_directions, 
    const float* triangles, 
    float *passed_test,
    int n_rays, 
    int n_triangles,
    cudaStream_t stream
) {
    watertightness_kernel<<<65536, 128, 0, stream>>>(
        ray_origins, 
        ray_directions, 
        triangles, 
        passed_test,
        n_rays, 
        n_triangles
    );
    


    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        throw std::runtime_error(
            Formatter() << "CUDA kernel failed : " << std::to_string(err)
        );
    }
}
