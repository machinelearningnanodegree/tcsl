FROM jupyter/scipy-notebook
RUN mkdir results
COPY app.py .
COPY student-data.csv .
