# DeepGlobe-Land-Cover-Classification-with-DeepLabV3-
Developed a deep learning model using DeepLabV3+ for classifying land cover types from satellite images using the DeepGlobe dataset.

## Preparation Script
The file ./Deepglobe_land_cover_classification.ipynb is a data preparation script on Google Colab that handles various steps in preparing the DeepGlobe Land Cover Classification dataset. The steps include:

* Satellite Image Segmentation: This step involves splitting satellite images into smaller regions or segments to facilitate data analysis.

* Conversion to One-Hot Encoding Format: Class labels are converted into One-Hot encoding. This means that each class is represented by a binary vector where only the position corresponding to the class is "1," and all others are "0." This allows the use of classification models.

* Inverse Decoding of One-Hot Encoding Representation: After training a model, the class predictions in One-Hot form are converted back to a more understandable representation, where each pixel of the image has a specific class label.

* Applying Color Encoding on Image Segmentation Results: The results of segmentation (or predictions) are often color-coded for better visualization. Each class is assigned a specific color, making the results more intuitive and easier to understand.

* Loading Satellite Images and Their Associated Segmentation Masks: This step loads the satellite images and their associated segmentation masks. Masks are images where each pixel corresponds to a specific class label for each region of the satellite image.

* Visualization of an Image and Its Mask: Once the images and masks are loaded, this step allows for the visualization of the satellite image and its corresponding segmentation mask to verify the alignment between the data and the labels.

* Defining Data Augmentations: Data augmentation involves applying various transformations (such as rotation, scaling, etc.) to images to generate new variations of the training data. This helps improve the robustness of the model.

* Visualizing Applied Augmentations: After defining the augmentations, this step involves displaying the modified images to ensure that the transformations have been correctly applied and that the augmented data is ready for model training.

These steps form a data preparation pipeline designed to make the images and their annotations ready for training an image segmentation model.
