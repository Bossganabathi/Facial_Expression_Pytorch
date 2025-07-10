# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# set the working directory
WORKDIR /app

# copy the current directory contains into the container at /app
COPY . /app

# install the required packages
RUN pip install --no-cache-dir -r requirements.txt

#make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]