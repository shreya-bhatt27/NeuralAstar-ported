ARG BASE_IMAGE_REPO
ARG BASE_IMAGE_TAG
ARG GPU_BASE_IMAGE_NAME
ARG LIGHTGBM_VERSION
ARG TORCH_VERSION

{{ if eq .Accelerator "gpu" }}
FROM gcr.io/kaggle-images/python-torch-whl:${GPU_BASE_IMAGE_NAME}-${BASE_IMAGE_TAG}-${TORCH_VERSION} AS torch_whl
FROM ${BASE_IMAGE_REPO}/${GPU_BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}
ENV CUDA_MAJOR_VERSION=11
ENV CUDA_MINOR_VERSION=0
# NVIDIA binaries from the host are mounted to /opt/bin.
ENV PATH=/opt/bin:${PATH}
# Add CUDA stubs to LD_LIBRARY_PATH to support building the GPU image on a CPU machine.
ENV LD_LIBRARY_PATH_NO_STUBS="$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs"
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
{{ else }}
FROM ${BASE_IMAGE_REPO}/${CPU_BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}
{{ end }}

#Install 
{{ if eq .Accelerator "gpu" }}
RUN conda install cudf=21.10 cuml=21.10 cudatoolkit=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION && \
    /tmp/clean-layer.sh
{{ end }}

# Install PyTorch
{{ if eq .Accelerator "gpu" }}
COPY --from=torch_whl /tmp/whl/*.whl /tmp/torch/
RUN pip install /tmp/torch/*.whl && \
    rm -rf /tmp/torch && \
    /tmp/clean-layer.sh
{{else}}
RUN pip install torch==$TORCH_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    /tmp/clean-layer.sh
{{ end }}

# Install GPU specific packages
{{ if eq .Accelerator "gpu" }}
# Install GPU-only packages
RUN pip install pycuda && \
    pip install pynvrtc && \
    pip install pynvml && \
    pip install nnabla-ext-cuda$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION && \
    /tmp/clean-layer.sh
{{ end }}
RUN pip install segmentation_models_pytorch && \
    pip install timm && \
    pip install subprocess && \
    pip install pytorch_lightning && \
    pip install wandb && \
    pip install bootstrapped && \
    pip install numpy && \
    pip install matplotlib && \
    pip install urllib3 && \
    pip install ipympl==0.7.0 && \
    /tmp/clean-layer.sh

#Setting backend fot Matplotlib
ENV MPLBACKEND "agg"

{{ if eq .Accelerator "gpu" }}
# Remove the CUDA stubs.
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH_NO_STUBS"
{{ end }}
