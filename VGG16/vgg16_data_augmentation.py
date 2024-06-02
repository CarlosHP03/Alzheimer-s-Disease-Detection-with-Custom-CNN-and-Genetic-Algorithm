from image_data_handler import ImageDataHandler
from model import PreTrainedClassifier

# Define data paths, hyperparameters
EPOCHS = 15
TARGET_SIZE = (299, 299)
BATCH_SIZE = 32
NUM_CLASSES = 2
MODEL = "VGG16"
ACTIVATION_FUNCTION = "sigmoid"
AUGMENT = True
CLASS_WEIGHTS = False

# Paths where the train images, test images and validation images are.
TRAIN_PATH = "../../dataset/train"
TEST_PATH = "../../dataset/test"
VAL_PATH = "../../dataset/val"


def main():
    data_handler = ImageDataHandler(TARGET_SIZE, BATCH_SIZE)

    # Load training, validation, and test data
    train_generator = data_handler.load_data(TRAIN_PATH, shuffle=AUGMENT)
    valid_generator = data_handler.load_data(VAL_PATH, shuffle=False)
    test_generator = data_handler.load_data(TEST_PATH, test=True, shuffle=False)

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = valid_generator.n // valid_generator.batch_size
    step_size_test = test_generator.n // test_generator.batch_size

    # Create the image classifier object
    model = PreTrainedClassifier(TARGET_SIZE, num_classes=NUM_CLASSES, activation=ACTIVATION_FUNCTION, model_name=MODEL,
                                 class_weight=CLASS_WEIGHTS, augment=AUGMENT)

    # Train the model
    history = model.train(train_generator, valid_generator, EPOCHS, step_size_train, step_size_valid, )

    model.evaluate(valid_generator, step_size_valid)

    model.plot_training_performance(history)

    test_generator.reset()
    prediction = model.test(test_generator, step_size_test)

    model.evaluation_metrics(prediction, test_generator)


if __name__ == '__main__':
    main()