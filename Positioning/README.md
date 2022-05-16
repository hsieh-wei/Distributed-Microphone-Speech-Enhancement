# Overview
The main job of this folder is to convert the information received by the millimeter wave radar into coordinates.

# fhistdataTOP.m

The fhistdata"number".mat is the information received by the millimeter wave radar, which can be converted into a series of continuous coordinates by fhistdataTOP.m

# position.m

The position"number".mat is the result of fhistdataTOP.m, we manually grab frames at different positions, and use position.m to get the coordinates of the position

# concatenate.m

The coordinate "number".mat is the result of position.m, we use concatenate.m to combine all the data to make it suitable for DDAE_Model