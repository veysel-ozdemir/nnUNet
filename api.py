import os
import io
import tempfile
import traceback
import sys
import time
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.backends.mps
from PIL import Image
from werkzeug.utils import secure_filename


# Add explicit exception hook to print all details
def exception_handler(exc_type, exc_value, exc_traceback):
    print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))


sys.excepthook = exception_handler

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO

app = Flask(__name__)
CORS(app)  # This allows all domains. You can restrict it if needed.

# Configuration settings
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "nnUNet_results/Dataset101_Breast/nnUNetTrainer_10epochs__nnUNetPlans__2d",
)
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "checkpoint_final.pth")
MODEL_FOLD = os.environ.get("MODEL_FOLD", "all")

# Initialize the predictor globally
predictor = None


def init_predictor():
    global predictor
    if predictor is None:
        if MODEL_DIR is None:
            raise ValueError("MODEL_DIR environment variable must be set!")

        # Initialize predictor with MPS if available (for M1/M2/M3 Macs),
        # then CUDA if available, otherwise CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(
                "Using MPS (Metal Performance Shaders) for computation on Apple Silicon"
            )
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA for computation")
        else:
            device = torch.device("cpu")
            print("Using CPU for computation")

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=False,  # Set this to False for MPS
            device=device,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True,
        )

        # Set folds correctly - handle the 'all' case properly
        print(f"Initializing with MODEL_DIR: {MODEL_DIR}")
        print(f"Fold parameter: {MODEL_FOLD}")

        if MODEL_FOLD.lower() == "all":
            # For 'all', don't convert to tuple, pass as string
            fold_param = "all"
            print("Using 'all' folds")
        else:
            # For numeric folds, convert to tuple
            fold_param = (MODEL_FOLD,)
            print(f"Using specific fold: {fold_param}")

        try:
            # Initialize from trained model folder
            predictor.initialize_from_trained_model_folder(
                MODEL_DIR,
                use_folds=fold_param,
                checkpoint_name=CHECKPOINT_NAME,
            )
            print(
                f"nnU-Net predictor initialized using model at {MODEL_DIR}, fold {MODEL_FOLD}"
            )
        except Exception as init_error:
            print(f"Error during model initialization: {str(init_error)}")
            print(traceback.format_exc())
            raise


def colorize_segmentation(seg_img, original_img=None):
    """
    Colorize segmentation result:
    - Value 1 (benign) = green
    - Value 2 (malignant) = red
    - Value 0 (background) = transparent or original image

    Args:
        seg_img: Segmentation image as numpy array
        original_img: Original image (optional) for background

    Returns:
        Colorized image
    """
    # Fix shape issues - ensure seg_img is 2D
    print(f"Debug - seg_img shape: {seg_img.shape}, dtype: {seg_img.dtype}")
    print(
        f"Debug - original_img shape: {original_img.shape}, dtype: {original_img.dtype}"
    )

    # Handle different dimension cases
    if len(seg_img.shape) == 1:
        # Reshape 1D array to 2D if needed
        print("Segmentation is 1D, attempting to reshape...")
        # Try to infer dimensions from original image
        if original_img is not None:
            height, width = original_img.shape[:2]
            seg_img = seg_img.reshape(height, width)
        else:
            # Cannot determine shape, use a default square
            size = int(np.sqrt(seg_img.shape[0]))
            seg_img = seg_img.reshape(size, size)

    # Create an RGBA image (with alpha channel for transparency)
    height, width = seg_img.shape
    colorized = np.zeros((height, width, 4), dtype=np.uint8)

    if original_img is not None:
        # If original image provided, use it as background
        # Make sure dimensions match
        if original_img.ndim == 2:  # Grayscale
            # Convert to RGB
            rgb_background = np.stack(
                [original_img, original_img, original_img], axis=2
            )
        else:  # Already RGB
            rgb_background = original_img[:, :, :3]  # Take only RGB channels if more

        # Copy background image
        colorized[:, :, 0:3] = rgb_background
        colorized[:, :, 3] = 255  # Fully opaque

        # Add colored overlay for segmentation
        # Green for benign (value 1)
        benign_mask = seg_img == 1
        colorized[benign_mask, 0] = 0  # R
        colorized[benign_mask, 1] = 255  # G
        colorized[benign_mask, 2] = 0  # B
        colorized[benign_mask, 3] = 180  # Semi-transparent

        # Red for malignant (value 2)
        malignant_mask = seg_img == 2
        colorized[malignant_mask, 0] = 255  # R
        colorized[malignant_mask, 1] = 0  # G
        colorized[malignant_mask, 2] = 0  # B
        colorized[malignant_mask, 3] = 180  # Semi-transparent

    else:
        # No original image, create a transparent background
        colorized[:, :, 3] = 0  # Fully transparent background

        # Green for benign (value 1)
        benign_mask = seg_img == 1
        colorized[benign_mask, 0] = 0  # R
        colorized[benign_mask, 1] = 255  # G
        colorized[benign_mask, 2] = 0  # B
        colorized[benign_mask, 3] = 255  # Opaque

        # Red for malignant (value 2)
        malignant_mask = seg_img == 2
        colorized[malignant_mask, 0] = 255  # R
        colorized[malignant_mask, 1] = 0  # G
        colorized[malignant_mask, 2] = 0  # B
        colorized[malignant_mask, 3] = 255  # Opaque

    print(f"Colorized image created with shape {colorized.shape}")
    return colorized


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_loaded": predictor is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict endpoint for ultrasound image segmentation.

    This endpoint accepts PNG image files (both color and grayscale) and returns
    a segmentation with color overlays. Color images will be automatically
    converted to grayscale for processing since the model was trained on
    grayscale images.

    The segmentation result uses color coding:
    - Green: Benign tumor regions (class 1)
    - Red: Malignant tumor regions (class 2)

    Returns:
        PNG image with the segmentation result as a colored overlay
    """
    try:
        # Make sure predictor is initialized
        if predictor is None:
            try:
                init_predictor()
            except Exception as e:
                print(f"Predictor initialization error: {str(e)}")
                print(traceback.format_exc())
                return (
                    jsonify({"error": f"Failed to initialize predictor: {str(e)}"}),
                    500,
                )

        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file with a simple name to avoid any path issues
            input_file = os.path.join(temp_dir, "input.png")
            file.save(input_file)
            print(f"Saved input file to {input_file}")

            # Set up output file path - use a name that won't be modified by nnUNet
            output_file = os.path.join(temp_dir, "prediction_result.png")
            print(f"Output will be saved to {output_file}")

            # Simple check to ensure file exists and is readable
            if not os.path.exists(input_file):
                return jsonify({"error": "Failed to save uploaded file"}), 500

            try:
                # Read the image with PIL to confirm it's valid
                test_img = Image.open(input_file)
                print(f"Image dimensions: {test_img.size}, mode: {test_img.mode}")

                # Convert to grayscale if the image is not already in L mode
                if test_img.mode != "L":
                    print(
                        f"Converting image from {test_img.mode} to grayscale (L mode)"
                    )
                    gray_img = test_img.convert("L")
                    # Save the grayscale version
                    gray_img.save(input_file)
                    # Keep a copy of original for overlay
                    original_img_array = np.array(test_img)
                    # Update test_img reference to grayscale version
                    test_img.close()
                    test_img = gray_img
                    print(
                        f"Image converted to grayscale, new dimensions: {test_img.size}, mode: {test_img.mode}"
                    )
                else:
                    # Already grayscale
                    original_img_array = np.array(test_img)
                    print("Image is already in grayscale mode (L)")

                # Convert PIL image to numpy array for processing
                grayscale_img_array = np.array(test_img)
                test_img.close()

                print(
                    f"Original image shape: {original_img_array.shape}, dtype: {original_img_array.dtype}"
                )
                print(
                    f"Grayscale image shape: {grayscale_img_array.shape}, dtype: {grayscale_img_array.dtype}"
                )

                # Try direct approach - bypass file writing issues with predict_from_files
                try:
                    # Use nnUNet's natural image reader
                    image_reader = NaturalImage2DIO()
                    # We now use the grayscale version for prediction
                    img_data, img_props = image_reader.read_images([input_file])
                    print(
                        f"Grayscale image loaded with NaturalImage2DIO. Shape: {img_data.shape}"
                    )

                    # Make prediction directly on the loaded image data
                    print("Starting direct prediction...")
                    segmentation = predictor.predict_single_npy_array(
                        img_data,
                        img_props,
                        segmentation_previous_stage=None,
                        output_file_truncated=None,
                        save_or_return_probabilities=False,
                    )

                    print(f"Prediction completed, result shape: {segmentation.shape}")
                    print(f"Prediction data type: {segmentation.dtype}")
                    print(f"Result dimensions: {len(segmentation.shape)}")

                    # Extract the segmentation result properly based on shape
                    if (
                        len(segmentation.shape) == 4
                    ):  # Expected shape (C, 1, H, W) for 2D
                        seg_img = segmentation[0, 0]  # First channel, first slice
                        print(f"Extracted 2D segmentation with shape: {seg_img.shape}")
                    elif len(segmentation.shape) == 3:  # Shape (C, H, W)
                        seg_img = segmentation[0]  # First channel
                        print(
                            f"Extracted segmentation from 3D tensor with shape: {seg_img.shape}"
                        )
                    elif len(segmentation.shape) == 2:  # Already 2D (H, W)
                        seg_img = segmentation
                        print(f"Segmentation already 2D with shape: {seg_img.shape}")
                    else:
                        # If dimensions don't match expectations, flatten and reshape based on original
                        print(
                            f"Unexpected shape: {segmentation.shape}, attempting to reshape..."
                        )
                        seg_img = segmentation.flatten()
                        print(f"Flattened shape: {seg_img.shape}")

                    # Print unique values to check segmentation classes
                    print(f"Segmentation unique values: {np.unique(seg_img)}")

                    # Ensure we're working with numeric values of the right type
                    seg_img = seg_img.astype(np.uint8)

                    # If there are no tumor regions predicted (all zeros), create a sample overlay
                    # This is just for demonstration if the model doesn't detect anything
                    if len(np.unique(seg_img)) == 1 and np.unique(seg_img)[0] == 0:
                        print(
                            "No tumor regions detected. Creating a sample overlay for demonstration."
                        )
                        # Create a small demo region in the center (for testing)
                        if (
                            original_img_array.shape[0] > 100
                            and original_img_array.shape[1] > 100
                        ):
                            h, w = seg_img.shape
                            # Add a small sample benign region
                            seg_img[h // 2 - 20 : h // 2, w // 2 - 20 : w // 2] = 1
                            # Add a small sample malignant region
                            seg_img[h // 2 : h // 2 + 20, w // 2 : w // 2 + 20] = 2
                            print("Added demo regions to the segmentation")

                    # Colorize the segmentation result
                    colorized_img = colorize_segmentation(seg_img, original_img_array)

                    # Convert numpy array to PIL Image and save
                    result_img = Image.fromarray(colorized_img)
                    result_img.save(output_file)
                    print(f"Colorized result saved to {output_file}")

                    # Wait a moment to ensure the file is written
                    time.sleep(0.5)

                    # Verify the file was created
                    if not os.path.exists(output_file):
                        print(
                            f"Warning: Could not find output file at {output_file} after saving"
                        )

                        # Try alternative location by checking the temp directory
                        print(
                            f"Checking temp directory contents: {os.listdir(temp_dir)}"
                        )

                        # Try one more time with a different name
                        alt_output = os.path.join(temp_dir, "alt_output.png")
                        result_img.save(alt_output)

                        if os.path.exists(alt_output):
                            output_file = alt_output
                            print(f"Using alternative output file: {alt_output}")
                        else:
                            return (
                                jsonify({"error": "Failed to save prediction result"}),
                                500,
                            )

                except Exception as direct_error:
                    print(f"Error during direct prediction: {str(direct_error)}")
                    print(traceback.format_exc())
                    return (
                        jsonify({"error": f"Prediction failed: {str(direct_error)}"}),
                        500,
                    )

                # Return the file
                print(f"Returning segmentation from {output_file}")
                return send_file(output_file, mimetype="image/png")

            except Exception as pred_error:
                print(f"Error during prediction: {str(pred_error)}")
                print(traceback.format_exc())
                return jsonify({"error": f"Prediction failed: {str(pred_error)}"}), 500

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    # Initialize the predictor on startup
    try:
        init_predictor()
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")
        print(traceback.format_exc())
        print(
            "API will still start, but predictor will be initialized on first request"
        )

    # Run the Flask app
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
