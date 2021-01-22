void watertightness(
    const float* ray_origins, 
    const float* ray_directions, 
    const float* triangles, 
    float *passed_test,
    int n_rays, 
    int n_triangles,
    cudaStream_t stream);

