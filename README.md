# Pytorch Color Model

A pytorch model used to discriminate dark and bright colors.

## Usage

To use the model, simply execute this command

````python
python model.py <red> <green> <blue>
````

For example:

````python
python model.py 0 0 255
````

## Calculation

This model has been trained with the following luminance equation:

````
Luminance = 0.299 * R + 0.587 * G + 0.114 * B
````