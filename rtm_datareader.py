import numpy
import onnxruntime as ort
import os
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image
from utils import *


def preprocess_data(img_folder: str, infer_size: int, pad_val: int, size_limit=0):
    img_files = os.listdir(img_folder)
    # shuffle
    img_files = numpy.random.permutation(img_files)
    if len(img_files) > size_limit:
        img_files = img_files[:size_limit]

    batch_data = []
    for file in img_files:
        img_path = os.path.join(img_folder, file)
        img = Image.open(img_path).convert('RGB')
        input_tensor, _, __, ___ = preprocess(img, max_size=infer_size, pad_val=pad_val)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        batch_data.append(input_tensor)

    batch_data = numpy.concatenate(numpy.expand_dims(batch_data, axis=0), axis=0)
    print(batch_data.shape)
    return batch_data


class RTMDetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, infer_size: int, pad_val: int, num_imgs: int):
        self.enum_data = None

        # Use inference session to get input name
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name

        # Convert image to input data
        self.batch_data = preprocess_data(calibration_image_folder, infer_size, pad_val, num_imgs)
        self.datasize = len(self.batch_data)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.batch_data]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None