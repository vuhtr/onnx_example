from onnxruntime.quantization import quantize_dynamic
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    args = parser.parse_args()
    return args

args = get_args()

model_fp32 = args.input_model
model_quant = args.output_model
quantized_model = quantize_dynamic(model_fp32, model_quant)