FROM padmec/pymoab-pytrilinos:3.6
WORKDIR /
COPY ./requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
