from image_data_handler import ImageDataHandler
from model import PreTrainedClassifier

# Define data paths, hyperparameters
EPOCHS = 15
TARGET_SIZE = (299, 299)
BATCH_SIZE = 32
MODEL = "InceptionV3"
AUGMENT = True
CLASS_WEIGHTS = True

# Paths where the train images, test images and validation images are.
TRAIN_PATH = "../../dataset/train"
TEST_PATH = "../../dataset/test"
VAL_PATH = "../../dataset/val"


data_handler = ImageDataHandler(TARGET_SIZE, BATCH_SIZE)

# Load training, validation, and test data
train_generator = data_handler.load_data(TRAIN_PATH, shuffle=AUGMENT)
valid_generator = data_handler.load_data(VAL_PATH, shuffle=False)
test_generator = data_handler.load_data(TEST_PATH, test=True, shuffle=False)

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

# Create the image classifier object
model = PreTrainedClassifier(TARGET_SIZE, num_classes=2, base_model_name=MODEL)

# Train the model
history = model.train(train_generator, valid_generator, EPOCHS, STEP_SIZE_TRAIN, STEP_SIZE_VALID,
                      class_weights=CLASS_WEIGHTS, augment=AUGMENT)

model.evaluate(valid_generator, STEP_SIZE_VALID)

model.plot_training_performance(history, class_weights=CLASS_WEIGHTS, augment=AUGMENT)

test_generator.reset()
prediction = model.test(test_generator, STEP_SIZE_TEST)

model.evaluation_metrics(prediction, test_generator, class_weights=CLASS_WEIGHTS, augment=AUGMENT)