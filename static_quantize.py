import argparse
import numpy as np
import onnxruntime
import time
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from rtm_datareader import RTMDetDataReader


def benchmark(model_path, infer_size):
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, infer_size, infer_size), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    # Benchmarking
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument("--calibrate_dir", default="./test_images", help="calibration data set")
    parser.add_argument("--calibrate_size", default=100, type=int, help="calibration data set size, 0 means all")
    parser.add_argument("--infer_size", default=800, type=int, help="inference size")
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    PAD_VAL = 114
    # MAX_LEN = 5

    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dir = args.calibrate_dir
    calibrate_size = args.calibrate_size
    infer_size = args.infer_size
    dr = RTMDetDataReader(calibration_dir, input_model_path, infer_size, pad_val=PAD_VAL, num_imgs=calibrate_size)

    # Calibrate and quantize model
    # Turn off model optimization during quantization
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        optimize_model=True,
        per_channel=args.per_channel,
        # per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path, infer_size)

    print("benchmarking int8 model...")
    benchmark(output_model_path, infer_size)


if __name__ == "__main__":
    main()