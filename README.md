# Distillation-Column-Soft-Sensor
Create, train and test a soft sensor to predict ethanol concentration of the output of a distillation column from other measured data (pressure, tray temperatures, flowrates, ...)

Distillation column dataset found on https://www.kaggle.com/datasets/jorgecote/distillation-column?resource=download (note that some values of the csv like 1.23e8 are written as 1,23e8, and therefore will cause errors unless you replace them with 1.23e8)

The model creation function can be used to quickly create and evaluate a model (saves both the evaluation metrics and the model), and can be used to easily study the effects of most of the training parameters on the model's performance, to then create the best-performing model

Follow the comments in the code for a walkthrough of the model's (and function's) development
If you use this code yourselves, make sure to have the csv in the same folder as the py file
