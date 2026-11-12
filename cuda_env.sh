# TF 2.12 için CUDA 11.8 / cuDNN 8.6 kütüphane yolu (venv dışında, .cuda11 altında).
# Kullanım: source cuda_env.sh
_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.cuda11/nvidia"
for s in cuda_runtime cuda_nvrtc cublas cudnn cufft curand cusolver cusparse; do
  export LD_LIBRARY_PATH="$_d/$s/lib:${LD_LIBRARY_PATH}"
done
