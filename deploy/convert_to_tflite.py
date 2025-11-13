import tensorflow as tf
import argparse
import os

def convert_to_tflite(model_path, output_dir, quantize=False):
    """
    Converts a saved Keras model (.h5) to a TensorFlow Lite model (.tflite).

    Args:
        model_path (str): Path to the input Keras model file.
        output_dir (str): Directory to save the .tflite model.
        quantize (bool): If True, applies default float16 quantization.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Load the Keras model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Keras model loaded successfully.")
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return

    # Create a TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply quantization if requested
    if quantize:
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        output_filename = os.path.splitext(os.path.basename(model_path))[0] + "_quant.tflite"
    else:
        output_filename = os.path.splitext(os.path.basename(model_path))[0] + ".tflite"

    # Convert the model
    try:
        tflite_model = converter.convert()
        print("Model converted to TFLite successfully.")
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        return

    # Save the TFLite model
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"\n✅ TFLite model saved to: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    # --- Latency Test ---
    test_latency(output_path)


def test_latency(tflite_model_path, num_runs=100):
    """
    Tests the inference latency of a TFLite model.

    Args:
        tflite_model_path (str): Path to the .tflite model file.
        num_runs (int): The number of inferences to run for averaging latency.
    """
    print("\n--- Testing TFLite Model Latency ---")
    
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create dummy input data matching the model's input shape
    input_shape = input_details[0]['shape']
    input_data = tf.random.uniform(input_shape, dtype=tf.float32)

    # Run inference multiple times to get an average latency
    latencies = []
    for _ in range(num_runs):
        start_time = tf.timestamp()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        end_time = tf.timestamp()
        latencies.append((end_time - start_time).numpy() * 1000) # Convert to ms

    avg_latency = sum(latencies) / num_runs
    print(f"Average latency over {num_runs} runs: {avg_latency:.4f} ms")
    print("Note: This is a rough estimate on your current machine.")


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite.")
    parser.add_argument("--model_path", type=str, default="models/saved_models/baseline_lstm.h5",
                        help="Path to the input Keras .h5 model.")
    parser.add_argument("--output_dir", type=str, default="deploy/tflite",
                        help="Directory to save the converted TFLite model.")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable float16 quantization to reduce model size.")
    
    args = parser.parse_args()

    convert_to_tflite(args.model_path, args.output_dir, args.quantize)
    
    print("\n--- Instructions for Raspberry Pi ---")
    print("1. Install TensorFlow Lite runtime: pip install tflite-runtime")
    print("2. Copy the generated .tflite file to your Raspberry Pi.")
    print("3. Use the TFLite runtime in your Python script to load the model and run inference.")


if __name__ == "__main__":
    # Example usage:
    # python deploy/convert_to_tflite.py --model_path models/saved_models/baseline_lstm.h5 --quantize
    main()
