FROM doduo2.umcn.nl/uokbaseimage/tensorflow_pytorch_python3:3

COPY --chown=user:user create_imgs_pdf.py /home/user/

RUN pip install --upgrade pip
RUN pip install --upgrade matplotlib
RUN pip install --upgrade numpy
RUN pip install --upgrade SimpleITK
RUN pip install --upgrade pandas
RUN pip install --upgrade openpyxl

WORKDIR /home/user
USER user

ENTRYPOINT ["python3.7", "-u", "create_imgs_pdf.py"]