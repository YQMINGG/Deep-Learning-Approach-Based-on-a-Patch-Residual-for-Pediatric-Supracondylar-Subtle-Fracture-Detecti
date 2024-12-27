# Deep Learning Approach Based on a Patch Residual for Pediatric Supracondylar Subtle Fracture Detection

# Introduction
Supracondylar humerus fractures in children are one of the most common elbow fractures in pediatrics, but their diagnosis is very challenging due to the anatomical features and imaging characteristics of the pediatric skeleton. In recent years, convolutional neural networks (CNNs) have achieved remarkable results in medical image analysis, but their performance usually relies on large-scale, high-quality labeled data. However, collecting supracondylar fractures samples is a time-consuming task, and the X-ray image features of lesion areas are difficult to be observed. To address this problem, this paper proposes a deep learning-based multiscale patch residual network (MPR) for automatic detection and localization of subtle pediatric supracondylar fractures. The MPR framework combines a convolutional neural network (CNN) for automatic feature extraction and a multiscale generative adversarial network (GAN) for learning skeletal integrity from healthy samples. The detection sensitivity of subtle fractures is enhanced by combining weighted binary cross-entropy loss (W-BCE). The method uses health images to model normal skeletal distribution, reducing the reliance on labeled fracture data and overcoming the challenges associated with limited pediatric datasets. On an independent test set, the model achieves 90.5% accuracy with 89% sensitivity, 92% specificity, and an F1 score of 0.906, exceeding the diagnostic performance of emergency medicine physicians and approaching that of pediatric radiologists. At the same time, the model possesses a fast inference speed (1.1 s/sheet), which provides potential for clinical applications.
![image](https://github.com/YQMINGG/Fracture_Detections/blob/master/Framediagram.png)


# Download the Model
You can download the model from the following link:
[https://huggingface.co/openai/diffusion-model-example](https://huggingface.co/YMING222/FractionDetection/tree/main)

and save it to :
./Weights_file/ 

# Detection
In order to examine the incoming X-ray images, run：

    python detect.py

