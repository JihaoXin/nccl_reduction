#include "reduction.hh"

extern const int num_elem;
extern int thread_per_block;
extern int num_block;
extern int* dev_num_elem;
#define max_interval 64.0 //63+1 largest_interval + ⌈log(n)⌉
using namespace std;
__global__ void cuda_quantize(
    float *dev_gradient,
    uint8_t *dev_compressed,
    const float*  dev_rand,
    float *dev_global_norm,
    int* dev_num_elem) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < *dev_num_elem; i += stride) {
     // Normalize -> [-1,1]
    dev_gradient[i] = dev_gradient[i]/(*dev_global_norm);
  
    // Decode
    int exp;
    float prob = abs(frexpf(dev_gradient[i], &exp)) / 0.5 - 1.; // exp = [-127, 1]; prob = [0.5, 1) -> [0, 1)
    if (dev_rand[i] >= prob) exp = exp - 1.0;// Prob < 1 so only round to 2^exp or 2^exp-1
    exp = max(exp, -63);
    exp = min(exp, 0); //exp = [-63,0]
    exp = -exp; //exp = [0,63]
    if (dev_gradient[i] < 0) exp += 128; // Negative Highest bit = 1
    if (dev_gradient[i] == 0) exp = 63;  // Set exp of 0 to -63
    dev_compressed[i] = static_cast<uint8_t>(exp);
  }
}

__global__ void cuda_dequantize(
    float *dev_gradient,
    uint8_t *dev_compressed,
    float *dev_global_norm,
    int* dev_num_elem) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < *dev_num_elem; i += stride) {
    // Decode
    int exp = dev_compressed[i];
    if(dev_compressed[i]>=128) exp -= 128;
    dev_gradient[i] = pow(2,-exp);
    if(exp>=63) dev_gradient[i] = 0;
    if(dev_compressed[i]>=128) dev_gradient[i] = -dev_gradient[i];
    // DeNormalize
    dev_gradient[i] = dev_gradient[i] * (*dev_global_norm);
  }
}

__global__ void cuda_reduce(
    uint8_t *dev_compressed_a,
    uint8_t *dev_compressed_b,
    float *dev_rand,
    int *dev_num_elem) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  // printf("ID %n reduced to %c", tid, dev_compressed_a[tid]);
  for (unsigned int i = tid; i < *dev_num_elem; i += stride) {
    // Decode
    int sign_a, sign_b, exp_a, exp_b;
    {
      if (dev_compressed_a[i] >= 128 ){
        sign_a = -1;
        exp_a = dev_compressed_a[i] - 128;
      } else{
        sign_a = 1;
        exp_a = dev_compressed_a[i];
      }
      if (dev_compressed_b[i] >= 128 ){
        sign_b = -1;
        exp_b = dev_compressed_b[i] - 128;
      } else{
        sign_b = 1;
        exp_b = dev_compressed_b[i];
      }
    }
    // Reduce
    int k = floor(log( 
            pow(2,-max_interval) +
            max(0.0 , dev_rand[i]-pow(2,-max_interval))
        ));
    int max_exp = max(exp_a, exp_b);
    int min_exp = min(exp_a, exp_b);
    int sign_ab = sign_a * sign_b;
    int diff = min_exp - max_exp + sign_ab - 1;
    int nonz = 1 - (exp_a == exp_b && sign_a != sign_b);
    int geq = (exp_a >= exp_b);
    int le = 1 - geq;
    int minz = (min_exp > 0);
    int reduce_sign = sign_a * geq + sign_b * le;
    int reduce_exp = nonz * (max_exp + minz * sign_ab * static_cast<int>(k<=diff));
    //Encode
    dev_compressed_a[i] = reduce_exp;
    if(reduce_sign == -1) dev_compressed_a[i] += 128;
    dev_compressed_b[i] = dev_compressed_a[i];
    
  }
}


// Quantize float32 gradient to uint_8 gradient
void quantize(float *dev_gradient, uint8_t *dev_compressed, float *dev_global_norm){
    float *dev_rand = dev_prob_generator();
    cuda_quantize<<<num_block, thread_per_block>>>(dev_gradient, dev_compressed, dev_rand, dev_global_norm, dev_num_elem);
    return;
}

// Aggregate two uint8 gradient
void reduce(uint8_t *dev_compressed_a, uint8_t *dev_compressed_b){
    float *dev_rand = dev_prob_generator();
    cuda_reduce<<<num_block, thread_per_block>>>(dev_compressed_a, dev_compressed_b, dev_rand, dev_num_elem);
    return;
}

// dequantize uint8 gradient to float32 gradient
void dequantize(float *dev_gradient, uint8_t *dev_compressed, float *dev_global_norm){
    cuda_dequantize<<<num_block, thread_per_block>>>(dev_gradient, dev_compressed,dev_global_norm, dev_num_elem);
    return;
}