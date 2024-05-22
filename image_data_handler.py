from keras.preprocessing.image import ImageDataGenerator


class ImageDataHandler:
    """
    This class handles loading and preprocessing image data for training, validation, and testing.
    """

    def __init__(self, target_size, batch_size, class_mode="categorical"):
        """
        Initializer for the ImageDataHandler class.

        Args:
            target_size: The target size for resizing images (e.g., (120, 120)).
            batch_size: The batch size for data generators.
            class_mode: The class mode for data generators (e.g., "categorical").
        """
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode

    def create_datagen(self, rescale=1./255., augment=True):
        """
        Creates an ImageDataGenerator object with specified parameters.

        Args:
            rescale: The value to rescale pixel values (default: 1./255.).
            augment: Whether to perform data augmentation (default: True).

        Returns:
            An ImageDataGenerator object.
        """
        datagen = ImageDataGenerator(rescale=rescale)
        if augment:
            datagen = ImageDataGenerator(rescale=rescale,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True)
        return datagen

    def load_data(self, data_path, test=False, shuffle=True):
        """
        Loads image data from a directory using a data generator.

        Args:
            data_path: The path to the directory containing images.
            shuffle: Whether to shuffle the data (default: True).
            test:.

        Returns:
            A Keras ImageDataGenerator object.
        """
        if test:
            datagen = self.create_datagen(augment=(shuffle is True))
            return datagen.flow_from_directory(
                directory=data_path,
                target_size=self.target_size,
                color_mode="rgb",
                batch_size=1,
                class_mode=self.class_mode,
                shuffle=shuffle,
                seed=42
            )
        else:
            datagen = self.create_datagen(augment=(shuffle is True))
            return datagen.flow_from_directory(
                directory=data_path,
                target_size=self.target_size,
                color_mode="rgb",
                batch_size=self.batch_size,
                class_mode=self.class_mode,
                shuffle=shuffle,
                seed=42
            )