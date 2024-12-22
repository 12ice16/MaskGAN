MaskGAN Model for Crack and Mask Generation:

Introduction:
This project implements an image processing system based on a custom Generative Adversarial Network (GAN) model, designed to automatically generate cracks and their masks from input images, while maintaining the background consistency outside the crack areas. With this model, users no longer need to manually annotate the crack regions. The system will automatically generate crack masks and ensure that the areas outside the cracks match the original image.

Key Features:
Automatic Crack and Mask Generation: The GAN model processes input images and automatically detects and generates crack regions along with their corresponding masks.
Background Consistency: The generated crack regions seamlessly blend with the background, ensuring that the areas outside the cracks remain identical to the original image.
No Manual Annotation Required: With a pretrained model, users don't need to manually annotate cracks or generate masks for each image.

Project Inspiration and Code References:
During the development of this project, we referenced portions of code from YOLOv5 and CycleGAN.
