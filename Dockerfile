# Use an official Python image as a base
FROM python:3.10

# Set the working directory in the container
WORKDIR /code

# Copy requirements file
COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire project
COPY ./app /code/app

# Command to run FastAPI using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
