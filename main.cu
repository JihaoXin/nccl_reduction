#include "reduction.cu"

using namespace std;
const int num_elem = 9;
int thread_per_block;
int num_block;
int* dev_num_elem;
const int MAX_THREAD_PER_BLOCK = 1024;
const int MAX_NUMBER_OF_BLOCK = 65535;
int main(){
    // Random initialize two uint8 array
    float *gradient_a = new float[num_elem];
    float *gradient_b = new float[num_elem];
    // float temp[num_elem] = {1,0.5,0.25,0.125,0, -0.125, -0.25, -0.5, -1};
    float temp[num_elem] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    float input_mean = 0;
    float input_std = 1.0;
    float global_norm = 0;
    default_random_engine generator(seed);
    normal_distribution<float> distribution(input_mean, input_std);
    for (int i = 0; i < num_elem; i++) {
        // gradient_a[i] = distribution(generator);
        // gradient_b[i] = distribution(generator);
        gradient_a[i] = temp[i];
        gradient_b[i] = temp[i];
        if(abs(gradient_a[i]) > global_norm) global_norm = abs(gradient_a[i]);
        if(abs(gradient_b[i]) > global_norm) global_norm = abs(gradient_b[i]);
    }

    // Allocate GPU Memory
    float *dev_gradient_a, *dev_gradient_b;
    float *dev_global_norm;
    uint8_t *dev_compressed_a, *dev_compressed_b;
    cudaMalloc(&dev_gradient_a, sizeof(float)*num_elem);
    cudaMalloc(&dev_compressed_a, sizeof(uint8_t)*num_elem);
    cudaMalloc(&dev_gradient_b, sizeof(float)*num_elem);
    cudaMalloc(&dev_compressed_b, sizeof(uint8_t)*num_elem);
    cudaMalloc(&dev_global_norm, sizeof(float));
    cudaMalloc(&dev_num_elem, sizeof(int));

    // Copy to GPU
    cudaMemcpy(dev_gradient_a, gradient_a, sizeof(float)*num_elem, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gradient_b, gradient_b, sizeof(float)*num_elem, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_global_norm, &global_norm, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_num_elem, &num_elem, sizeof(int), cudaMemcpyHostToDevice);
    // Threads & Blocks
    thread_per_block = min(MAX_THREAD_PER_BLOCK, num_elem);
    num_block = min( (num_elem + thread_per_block - 1)/thread_per_block, MAX_NUMBER_OF_BLOCK);
    assert(thread_per_block > 0); assert(num_block > 0);

    // Log
    cout<<"Input:"<<endl;
    cout<<"Distribution = Gaussian"<<"; Mean = "<<input_mean<<"; Std = "
        <<input_std<<"; Size = "<<num_elem<<"; Global Norm = "<<global_norm<<endl;
    cout<<"CUDA"<<endl;
    cout<<"Thread = "<<thread_per_block<<"; num_block = "<<num_block<<endl;

    // Quantization
    quantize(dev_gradient_a, dev_compressed_a, dev_global_norm);
    quantize(dev_gradient_b, dev_compressed_b,dev_global_norm);
    cudaStreamSynchronize(0);
    // Reduce
    reduce(dev_compressed_a, dev_compressed_b);

    // Decompress
    dequantize(dev_gradient_a, dev_compressed_a, dev_global_norm);
    dequantize(dev_gradient_b, dev_compressed_b, dev_global_norm);
    cudaStreamSynchronize(0);

    // Copy to CPU
    cudaMemcpy(gradient_a, dev_gradient_a, sizeof(float)*num_elem, cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient_b, dev_gradient_b, sizeof(float)*num_elem, cudaMemcpyDeviceToHost);

    uint8_t *compressed_a = new uint8_t[num_elem];
    int maximum = 0;
    cudaMemcpy(compressed_a, dev_compressed_a, sizeof(uint8_t)*num_elem, cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_elem; i++) {
        cout<<std::bitset<8>(compressed_a[i])<<" ";
        cout<<gradient_a[i]<<endl;
        // if(abs(gradient_a[i]) > maximum) maximum = abs(gradient_a[i]);
        // if(abs(gradient_b[i]) > maximum) maximum = abs(gradient_b[i]);
    }
    cout<<endl;
    // cout<<"Maximum is "<<maximum<<endl;
}