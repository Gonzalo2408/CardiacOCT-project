FROM {{baseimage}} as processor

RUN pip install astropy

# Copy model files into results
COPY --chown=user models ${RESULTS_FOLDER}

# Copy configuration file
COPY --chown=user config.json /home/user/processor

# Update process.py file for testing purposes
COPY --chown=user process.py /home/user/processor/process.py

# Update process_img.py file for testing purposes
COPY --chown=user process_img.py /home/user/processor/process_img.py

# Copy fonts
COPY --chown=user:user ./fonts /home/user/fonts/

# Enable/Disable alternative resampling
ENV DIAG_NNUNET_ALT_RESAMPLING {{DIAG_NNUNET_ALT_RESAMPLING}}
