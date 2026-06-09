FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git curl vim \
    gcc g++ gfortran make \
    libopenblas-dev liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# scipy, h5py, tensorboard
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PETSc must be present before petsc4py is installed so petsc4py links
# against it. --no-build-isolation prevents pip from creating a fresh
# build environment (where petsc would be absent) during petsc4py's install.
RUN pip install --no-cache-dir "petsc>=3.20" && \
    pip install --no-cache-dir "cython>=3.0.0" && \
    pip install --no-cache-dir --no-build-isolation "petsc4py>=3.20"

RUN mkdir -p /workspace/checkpoints /workspace/data /workspace/logs

CMD ["python", "train_solver_selector.py"]