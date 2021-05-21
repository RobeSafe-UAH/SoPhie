# SoPhie
SoPhie: An Attentive GAN for Predicting Paths Compliant to Social andPhysical Constraints

<img src="media/system_pipeline.svg"/>

## Overview
This paper addresses the problem of path prediction for multiple interacting agents in a scene, which is a crucial step for many autonomous platforms such as self-driving cars and social robots. We present SoPhie; an interpretableframework based on Generative Adversarial Network (GAN), which leverages two sources of information, the path history of all the agents in a scene, and the scene context information, using images of the scene. To predict a future path for an agent, both physical and social information must be leveraged. Previous work has not been successful to jointly model physical and social interactions. Our approach blends asocial attention mechanism with a physical attention thathelps the model to learn where to look in a large scene and extract the most salient parts of the image relevant to the path. 

<!-- Second, the system is validated ([Qualitative Results](https://cutt.ly/uk9ziaq)) in the CARLA simulator fulfilling the requirements of the Euro-NCAP evaluation for Unexpected Vulnerable Road Users (VRU), where a pedestrian suddenly jumps into the road and the vehicle has to avoid collision or reduce the impact velocity as much as possible. Finally, a comparison between our HD map based perception strategy and our previous work with rectangular based approach is carried out, demonstrating how incorporating enriched topological map information increases the reliability of the Autonomous Driving (AD) stack. Code is publicly available ([Code](https://github.com/Cram3r95/map-filtered-mot)) as a ROS package. -->

## Requirements

<!-- Note that due to ROS1 limitations (till Noetic version), specially in terms of TF ROS package, we limited the Python version to 2.7. Future works will integrate the code using ROS1 Noetic or ROS2, improving the version to Python3. -->

<!-- - Python3.8 
- Numpy
- ROS melodic
- HD map information (Monitorized lanes)
- scikit-image==0.17.2
- lap==0.4.0 -->
- OpenCV==4.1.1
- YAML
- ProDict
- torch (1.8.0+cu111)
- torchfile (0.1.0)
- torchsummary (1.5.1)
- torchtext (0.5.0)
- torchvision (0.9.0+cu111)

## Get Started and Usage
Coming soon ...
## Quantitative results
Coming soon ...
## Qualitative results
Coming soon ...

