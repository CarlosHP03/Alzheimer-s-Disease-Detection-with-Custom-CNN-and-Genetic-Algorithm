from image_data_handler import ImageDataHandler
from model import EnsembleModel


TARGET_SIZE = (299, 299)
BATCH_SIZE = 32
MODEL_NAME = "Ensemble_v2"
MODEL1_PATH = '../vgg16/models/vgg16_data_augmentation/vgg16_data_augmentation.h5'
MODEL2_PATH = '../inceptionv3/models/inceptionv3_data_augmentation/inceptionv3_data_augmentation.h5'
MODEL3_PATH = '../resnet50/models/resnet50_data_augmentation/resnet50_data_augmentation.h5'
TEST_PATH = "../../dataset/test"


def main():
    data_handler = ImageDataHandler(TARGET_SIZE, BATCH_SIZE)

    # Load training, validation, and test data
    test_generator = data_handler.load_data(TEST_PATH, test=True, shuffle=False)

    step_size_test = test_generator.n//test_generator.batch_size

    model = EnsembleModel(input_shape=TARGET_SIZE, model1_path=MODEL1_PATH,model2_path=MODEL2_PATH,
                          model3_path=MODEL3_PATH, model_name=MODEL_NAME)
    test_generator.reset()
    prediction = model.test(test_generator, step_size_test)

    model.evaluation_metrics(prediction, test_generator)


if __name__ == '__main__':
    main()
