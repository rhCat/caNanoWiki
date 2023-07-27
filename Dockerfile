# Use the official Ubuntu 20.04 image as the base
FROM ubuntu:20.04

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Add Miniconda to the PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Mamba
# RUN conda install -y -c conda-forge mamba && \
#     mamba --version

# Create a Conda environment using Mamba
COPY environment.yml /tmp/environment.yml
RUN conda env create -n caNanoWikiAI -f /tmp/environment.yml && \
    rm /tmp/environment.yml

# Activate the Conda environment by default
ENV PATH=$CONDA_DIR/envs/caNanoWikiAI/bin:$PATH

# Set the working directory in the container
WORKDIR /app

# Copy your application files
COPY . /app

# Expose the container port
EXPOSE 5000

# Set environment variables (optional)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Define the command to run your application
CMD [ "python", "app.py" ]