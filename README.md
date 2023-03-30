# Project for Deep Learning

Final project for the deep learning course. 


Recent advancements in deep learning techniques have shown promising results in medical image analysis and classification, particularly in ophthalmology. In the context of retinal image analysis, several studies have investigated the use of deep learning models for the detection and classification of
retinal diseases. For the project, we implemented all deep learning libraries and layers from scratch to build a multi-layered perception model and a CNN model. For the following study we created various unique networks and architectures to design our Convolutional Neural Network. We also Drew inference from various other architectures including a Multi layer perceptron model for multi-class classification problem.

More detailed analysis of how we approached this problem is given in the research paper present in the repository.

To run the code:

```bash
Python image_preprocessing_windows.py
```

To run with 100 samples run the following command:
```bash
Python rfmd_100.py
```

To train with all 949 samples, run the following command:

```bash
Python rfmd.py
```

To train with all 949 samples for the MLP model
```bash
Python mlp.py
```

Multi-labeled-classification-of-retina-image:

The data for the project is taken from a data challenge.: RFMD: https://riadd.grand-challenge.org/

Model  | Sample Size.  | Train Acc. |Validation Acc.|
------ | ------------- | -----------|---------------|
MLP1   | 949           |    85%     |       67%     |
CNN 1  | 949           |    83%     |       69%     |
-----------------------------------------------------

In the future, I plan to implement the same problem using advanced and faster approaches like using VGGNet model as a pre-trained model to classify multi-labeled images. This will be done using Tensorflow and Keras.

Project Group and Contribution:
1. Rishabh Sharma: Coding classes for different layers of the neural network. and implementing MLP model.
2. Ujjwal Malik: Coding classes for different layers of the neural network. and implementing CNN model.
3. Jasjiv Singh: Coding classes for different layers of the neural network. and implementing CNN model.

