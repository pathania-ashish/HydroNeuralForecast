FROM continuumio/miniconda3:latest

# Set environment name
ENV ENV_NAME=neuralforecast_3.10

# Create conda environment with Python 3.10 and PyTorch (CUDA 12.4)
RUN conda create -n ${ENV_NAME} python=3.10 -y && \
    conda run -n ${ENV_NAME} conda install -y \
        pytorch=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia && \
    conda clean -afy

# Set working directory
WORKDIR /app

# Copy requirements and install pip packages
COPY requirements.txt .
RUN conda run -n ${ENV_NAME} pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Install the neuralforecast package in editable mode
RUN conda run -n ${ENV_NAME} pip install -e .

# Activate the environment by default
RUN echo "conda activate ${ENV_NAME}" >> ~/.bashrc
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH

CMD ["bash"]
