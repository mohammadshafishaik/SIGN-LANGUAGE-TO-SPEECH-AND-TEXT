import coremltools as ct
import tensorflow as tf
import argparse
import os

def convert_to_coreml(model_path, output_dir, quantize=False):
    """
    Converts a saved Keras model (.h5) to a Core ML model (.mlmodel).

    Args:
        model_path (str): Path to the input Keras model file.
        output_dir (str): Directory to save the .mlmodel file.
        quantize (bool): If True, applies 16-bit float quantization.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load the Keras model
    try:
        keras_model = tf.keras.models.load_model(model_path)
        print("Keras model loaded successfully.")
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return

    # Define the input type for the Core ML model
    # This should match the input shape of your Keras model
    input_shape = keras_model.input_shape
    # Core ML expects input shape as (Seq, Batch, C, H, W) or similar
    # For our LSTM, it's (Batch, Seq, Features) -> we describe the sequence part
    input_features = [ct.TensorType(name="input_1", shape=input_shape)]

    # Convert the model
    print("Converting model to Core ML format...")
    try:
        # The conversion process depends on the TF version and coremltools version.
        # This is a common approach for TF2 models.
        mlmodel = ct.convert(
            keras_model,
            inputs=input_features,
            # classifier_config=ct.ClassifierConfig(class_labels), # If you have class labels
        )
        print("Model converted to Core ML successfully.")
    except Exception as e:
        print(f"Error during Core ML conversion: {e}")
        return

    # Apply quantization if requested
    if quantize:
        print("Applying FP16 quantization...")
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel, nbits=16
        )
        output_filename = os.path.splitext(os.path.basename(model_path))[0] + "_quant.mlmodel"
    else:
        output_filename = os.path.splitext(os.path.basename(model_path))[0] + ".mlmodel"

    # Save the Core ML model
    output_path = os.path.join(output_dir, output_filename)
    mlmodel.save(output_path)

    print(f"\n✅ Core ML model saved to: {output_path}")
    
    # --- Print model details ---
    print("\n--- Core ML Model Details ---")
    spec = mlmodel.get_spec()
    print(f"Model Description: {mlmodel.short_description}")
    print("\nInput description:")
    for i, input_desc in enumerate(spec.description.input):
        print(f"  [{i}] Name: {input_desc.name}")
        print(f"      Type: {input_desc.type}")
    
    print("\nOutput description:")
    for i, output_desc in enumerate(spec.description.output):
        print(f"  [{i}] Name: {output_desc.name}")
        print(f"      Type: {output_desc.type}")


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to Core ML.")
    parser.add_argument("--model_path", type=str, default="models/saved_models/baseline_lstm.h5",
                        help="Path to the input Keras .h5 model.")
    parser.add_argument("--output_dir", type=str, default="deploy/coreml",
                        help="Directory to save the converted Core ML model.")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable 16-bit weight quantization.")
    
    args = parser.parse_args()

    convert_to_coreml(args.model_path, args.output_dir, args.quantize)
    
    print("\n--- Instructions for iPhone/iOS ---")
    print("1. Drag the generated .mlmodel file into your Xcode project.")
    print("2. Ensure the model is added to your app's target.")
    print("3. Use Vision and Core ML frameworks in Swift to load the model and perform predictions on camera frames.")


if __name__ == "__main__":
    # Example usage:
    # python deploy/convert_to_coreml.py --model_path models/saved_models/baseline_lstm.h5 --quantize
    main()
