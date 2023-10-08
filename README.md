# Pytorch Tint Predictor

A pytorch model used to discriminate dark and bright colors.

## Usage

### Command line
To use the model, simply execute this command

````python
python model.py <red> <green> <blue>
````

For example:

````python
python model.py 0 0 255
````

### Web server
Alternatively, you can use the webserver contained in _deploy folder. First you'd have to run the server:

````python
python server.py
````

And then submit a POST request:

````curl
curl -d {"red": 255,"green": 0,"blue": 42} -X POST http://localhost:8080/predict
````

### Docker
I also deployed a container if you are using Docker

https://hub.docker.com/r/azenox/tint-predictor/tags

````docker
docker run -p 8080:8080 azenox/tint-predictor:latest
````

And then submit a POST request:

````curl
curl -d {"red": 255,"green": 0,"blue": 42} -X POST http://127.0.0.1:8080/predict
````

## Calculation

This model has been trained with the following luminance equation:

````
Luminance = 0.299 * R + 0.587 * G + 0.114 * B
````