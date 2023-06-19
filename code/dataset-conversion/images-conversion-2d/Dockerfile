FROM doduo2.umcn.nl/uokbaseimage/tensorflow_pytorch_python3:3

COPY --chown=user:user dicom_to_nifti_preprocessing_2d.py /home/user/

RUN pip install --upgrade pip
RUN pip install --upgrade numpy
RUN pip install --upgrade SimpleITK
RUN pip install --upgrade pandas
RUN pip install --upgrade openpyxl
RUN pip install --upgrade opencv-python

WORKDIR /home/user
USER user

ENTRYPOINT ["python3.7", "-u", "dicom_to_nifti_preprocessing_2d.py"]