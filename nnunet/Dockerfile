FROM doduo1.umcn.nl/uokbaseimage/diag:tf2.9-pt1.11-v1

# Configuration
RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "OMP_NUM_THREADS=1" >> /etc/environment

# Install a few dependencies that are not automatically installed, plus nnU-net
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/Gonzalo2408/nnUNet.git /home/user/nnunet && \
    git -C /home/user/nnunet checkout 1.7.0-6 && \
    chown -R user /home/user/nnunet && \
    pip3 install \
        -e /home/user/nnunet \
        graphviz \
        git+https://github.com/nanohanno/hiddenlayer.git@bugfix/get_trace_graph#egg=hiddenlayer \
        carbontracker && \
    rm -rf ~/.cache/pip

# Configure entrypoint
WORKDIR /home/user
USER root

COPY --chown=user nnunet_wrapper.py /home/user/nnunet_wrapper.py
RUN chmod +x /home/user/nnunet_wrapper.py && \
    ln -s /home/user/nnunet_wrapper.py /usr/local/bin/nnunet

#ENTRYPOINT ["python3", "-u", "nnunet_wrapper.py"]

ENTRYPOINT ["/bin/bash"]


