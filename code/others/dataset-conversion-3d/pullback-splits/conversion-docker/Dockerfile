FROM doduo2.umcn.nl/uokbaseimage/tensorflow_pytorch_python3:3

COPY --chown=user:user 3d_dicom_sampling_grayscale.py /home/user/

RUN pip install --upgrade pip
RUN pip install --upgrade numpy
RUN pip install --upgrade SimpleITK
RUN pip install --upgrade pandas
RUN pip install --upgrade openpyxl
RUN pip install --upgrade scikit-image


WORKDIR /home/user

USER user

ENTRYPOINT ["python3.7", "-u", "3d_dicom_sampling_grayscale.py"]
