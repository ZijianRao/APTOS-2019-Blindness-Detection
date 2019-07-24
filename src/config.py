PREV_DATA_PATH = '../data/previous/trainLabels15.csv'
PREV_TEST_PATH = '../data/previous/testLabels15.csv'
CLEAN_TRAIN_DATA_PATH = '../data/clean_sample.csv'
TRAIN_DATA_PATH = '../data/train.csv'
TRAIN_IMAGE_PATH = '../data/train_images/{}.png'
PREV_TRAIN_IMAGE_PATH = '../data/previous/resized_train_15/{}.jpg'
PREV_TEST_IMAGE_PATH = '../data/previous/resized_test_15/{}.jpg'

LOCAL_TRAIN_DATA_PATH = '../data/train.csv'
LOCAL_CLEAN_TRAIN_DATA_PATH = '../data/clean_sample.csv'
LOCAL_TRAIN_IMAGE_PATH = '../data/train_images'
LOCAL_TRAIN_IMAGE_ARRAY_PATH = '../data/train_{}.p'
LOCAL_CLEAN_TRAIN_IMAGE_ARRAY_PATH = '../data/train_clean_{}.p'
REMOTE_TRAIN_DATA_PATH = '../data/train.csv'
REMOTE_TRAIN_IMAGE_PATH = '../data/train_images'

LOCAL_TEST_DATA_PATH = '../data/test.csv'
LOCAL_TEST_IMAGE_PATH = '../data/test_images'
REMOTE_TEST_DATA_PATH = '../data/train.csv'
REMOTE_TEST_IMAGE_PATH = '../data/train_images'

DATA_PATH = '../data/'
NUM_WORKERS = 0
BUCKET_SPLIT = 15

VALIDATION_SIZE = 0.2
BATCH_SIZE = 20
IMG_SIZE = 224
NORMALIZE = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# NORMALIZE = None
# CUTOFF_COEF = [0.57, 1.37, 2.57, 3.57]
CUTOFF_COEF = [0.5, 1.5, 2.5, 3.5]

PRETRAINED_PATH = {'resnet50': '../../pytorch-pretrained-models/resnet50-19c8e357.pth',
                   'resnet101': '../../pytorch-pretrained-models/resnet101-5d3b4d8f.pth',
                   'resnext101_32x8d': '../../pytorch-pretrained-models/ig_resnext101_32x8-c38310e5.pth'
                   }
CHECKOUT_PATH = '../data/check_out/'
LOG_PATH = '../data/log/'