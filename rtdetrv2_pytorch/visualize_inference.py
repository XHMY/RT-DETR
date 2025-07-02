#!/usr/bin/env python3
"""
Inference Visualization Script for RT-DETR
Generate inference visualization results on validation set with bounding boxes.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, 'src')

from src.core import YAMLConfig
from src.misc import dist_utils


def draw_predictions(image_pil, labels, boxes, scores, class_names=None, threshold=0.3, save_path=None):
    """
    Draw predictions on PIL image and save.
    
    Args:
        image_pil: PIL Image
        labels: torch.Tensor of shape [N] with class labels
        boxes: torch.Tensor of shape [N, 4] with bounding boxes in xyxy format
        scores: torch.Tensor of shape [N] with confidence scores
        class_names: List of class names (optional)
        threshold: Confidence threshold for displaying boxes
        save_path: Path to save the visualized image
    """
    # Filter predictions by threshold
    mask = scores > threshold
    if not mask.any():
        # No predictions above threshold, save original image
        if save_path:
            image_pil.save(save_path, quality=95, optimize=True)
        return image_pil
    
    filtered_labels = labels[mask]
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    
    # Create a copy for drawing
    image_draw = image_pil.copy()
    draw = ImageDraw.Draw(image_draw)
    
    # Calculate adaptive font size based on image dimensions
    img_width, img_height = image_pil.size
    font_size = max(12, min(24, int(min(img_width, img_height) / 40)))
    
    # Try to load a font with adaptive size, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Calculate adaptive line width based on image size
    line_width = max(2, int(min(img_width, img_height) / 200))
    
    # Color palette for different classes
    colors = [
        'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown',
        'gray', 'cyan', 'magenta', 'lime', 'indigo', 'violet', 'turquoise'
    ] * 10  # Repeat colors if we have more than 15 classes
    
    for i, (label, box, score) in enumerate(zip(filtered_labels, filtered_boxes, filtered_scores)):
        x1, y1, x2, y2 = box.cpu().numpy()
        label_idx = label.cpu().item()
        score_val = score.cpu().item()
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Skip if box is too small or invalid
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Choose color based on class
        color = colors[label_idx % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Prepare text
        if class_names and label_idx < len(class_names):
            text = f"{class_names[label_idx]}: {score_val:.2f}"
        else:
            text = f"Class {label_idx}: {score_val:.2f}"
        
        # Calculate text position - place above the box if there's space, otherwise inside
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Position text above the box if there's space, otherwise inside at the top
        if y1 - text_height - 2 >= 0:
            text_x = x1
            text_y = y1 - text_height - 2
        else:
            text_x = x1 + 2
            text_y = y1 + 2
        
        # Ensure text doesn't go outside image bounds
        text_x = max(0, min(text_x, img_width - text_width))
        text_y = max(0, min(text_y, img_height - text_height))
        
        # Draw text background rectangle
        padding = 2
        bg_bbox = [
            text_x - padding, 
            text_y - padding, 
            text_x + text_width + padding, 
            text_y + text_height + padding
        ]
        draw.rectangle(bg_bbox, fill=color)
        
        # Draw text
        draw.text((text_x, text_y), text, fill='white', font=font)
    
    # Save the image with high quality to preserve details
    if save_path:
        image_draw.save(save_path, quality=95, optimize=True)
        
    return image_draw


def load_model_and_checkpoint(config_path, checkpoint_path, device):
    """
    Load model and checkpoint.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model in eval mode
        postprocessor: Postprocessor for model outputs
        cfg: Configuration object
    """
    print(f"Loading config from: {config_path}")
    cfg = YAMLConfig(config_path)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load state dict from checkpoint
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']['module']
        print("Using EMA weights")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Using model weights")
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint
        print("Using checkpoint as state dict")
    
    # Load model state
    cfg.model.load_state_dict(state_dict, strict=False)
    
    # Create deployment model
    class InferenceModel(nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model.deploy() if hasattr(model, 'deploy') else model
            self.postprocessor = postprocessor.deploy() if hasattr(postprocessor, 'deploy') else postprocessor
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            results = self.postprocessor(outputs, orig_target_sizes)
            return results
    
    model = InferenceModel(cfg.model, cfg.postprocessor).to(device)
    model.eval()
    
    return model, cfg


def create_transforms(input_size=640):
    """Create image transforms for inference."""
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])


def get_class_names(num_classes):
    """Get class names for visualization based on the actual stomach detection dataset."""
    # Based on mscoco_category2name from coco_dataset.py, translated to English
    # The mapping in coco_dataset.py uses category IDs: 0, 4, 5, 8, 10, 14, 17, 22, 23, 24, 26, 29, 40, 46
    # But mscoco_category2label maps them to sequential label indices 0-13
    stomach_classes_english = [
        "Barrett's Esophagus",           # 0: "Barrett食管"
        "Reflux Esophagitis",            # 4: "反流性食管炎" 
        "Right Arytenoid Cartilage",     # 5: "右侧杓状软骨"
        "Glottis",                       # 8: "声门"
        "Left Arytenoid Cartilage",      # 10: "左侧杓状软骨"
        "Fungal Esophagitis",            # 14: "真菌性食管炎"
        "Mucosal Folds",                 # 17: "襞裂"
        "Esophageal Papilloma",          # 22: "食管乳头状瘤"
        "Esophageal Cyst",               # 23: "食管囊肿"
        "Esophageal Leiomyoma",          # 24: "食管平滑肌瘤"
        "Esophageal Foreign Body",       # 26: "食管异物"
        "Esophageal Diverticulum",       # 29: "食管憩室"
        "Esophageal Glycogenic Acanthosis",  # 40: "食管糖原棘皮症"
        "Esophageal Varices"             # 46: "食管静脉曲张"
    ]
    
    if num_classes <= len(stomach_classes_english):
        return stomach_classes_english[:num_classes]
    else:
        # If somehow we have more classes than expected, extend with generic names
        extended_classes = stomach_classes_english.copy()
        for i in range(len(stomach_classes_english), num_classes):
            extended_classes.append(f'Class_{i}')
        return extended_classes


def main():
    parser = argparse.ArgumentParser(description='Generate inference visualization on validation set')
    parser.add_argument('--config', '-c', type=str,
                        default='configs/rtdetrv2/rtdetrv2_r101vd_6x_0528stomach.yml',
                        help='Path to model config file')
    parser.add_argument('--checkpoint', '-ckpt', type=str,
                        default='output/rtdetrv2_r101vd_6x_0528stomach_ft_0603/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', '-o', type=str, 
                        default='output/rtdetrv2_r101vd_6x_0528stomach_ft_0603/visualization',
                        help='Output directory for visualization results')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device to run inference on')
    parser.add_argument('--input_size', '-s', type=int, default=640,
                        help='Input image size for model')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='Confidence threshold for predictions')
    parser.add_argument('--max_images', '-m', type=int, default=None,
                        help='Maximum number of images to process (None for all)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and checkpoint
    model, cfg = load_model_and_checkpoint(args.config, args.checkpoint, device)
    
    # Setup validation dataloader
    val_dataloader = cfg.val_dataloader
    print(f"Validation dataset size: {len(val_dataloader.dataset)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualization results to: {output_dir}")
    
    # Get class names
    num_classes = getattr(cfg, 'num_classes', 14)  # Default to 14 for stomach dataset
    class_names = get_class_names(num_classes)
    print(f"Number of classes: {num_classes}")
    
    # Create transforms
    transforms = create_transforms(args.input_size)
    
    # Run inference and visualization
    print("Starting inference and visualization...")
    
    processed_count = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(tqdm(val_dataloader, desc="Processing")):
            if args.max_images and processed_count >= args.max_images:
                break
                
            # Process each image in the batch
            for i, (sample, target) in enumerate(zip(samples, targets)):
                if args.max_images and processed_count >= args.max_images:
                    break
                
                # Get original image info
                image_id = target.get('image_id', f'image_{batch_idx}_{i}')
                orig_size = target['orig_size'].cpu().numpy()  # [width, height] from target
                orig_w, orig_h = orig_size
                
                # Load the original image directly from the dataset to preserve aspect ratio
                try:
                    # Get the image path from the dataset
                    dataset = val_dataloader.dataset
                    img_folder = dataset.img_folder
                    
                    # Get the image filename from COCO dataset
                    if hasattr(dataset, 'coco') and hasattr(dataset.coco, 'imgs'):
                        img_info = dataset.coco.imgs[image_id.item() if isinstance(image_id, torch.Tensor) else image_id]
                        img_filename = img_info['file_name']
                        img_path = os.path.join(img_folder, img_filename)
                        
                        # Load original image
                        original_image = Image.open(img_path).convert('RGB')
                        print(f"Loaded original image: {img_path}, size: {original_image.size}")
                    else:
                        # Fallback: use the image from dataloader but try to preserve aspect ratio
                        if isinstance(sample, torch.Tensor):
                            # Convert from tensor format to PIL
                            sample_np = sample.permute(1, 2, 0).cpu().numpy()
                            if sample_np.max() <= 1.0:
                                sample_np = (sample_np * 255).astype(np.uint8)
                            else:
                                sample_np = sample_np.astype(np.uint8)
                            original_image = Image.fromarray(sample_np)
                        else:
                            original_image = sample
                        
                        # Resize to original dimensions if different
                        if original_image.size != (orig_w, orig_h):
                            original_image = original_image.resize((orig_w, orig_h), Image.LANCZOS)
                
                except Exception as e:
                    print(f"Warning: Could not load original image for ID {image_id}: {e}")
                    # Fallback to using sample from dataloader
                    if isinstance(sample, torch.Tensor):
                        sample_np = sample.permute(1, 2, 0).cpu().numpy()
                        if sample_np.max() <= 1.0:
                            sample_np = (sample_np * 255).astype(np.uint8)
                        else:
                            sample_np = sample_np.astype(np.uint8)
                        original_image = Image.fromarray(sample_np)
                    else:
                        original_image = sample
                    
                    # Resize to original dimensions if different
                    if original_image.size != (orig_w, orig_h):
                        original_image = original_image.resize((orig_w, orig_h), Image.LANCZOS)
                
                # Prepare input for model inference (separate from visualization image)
                model_input = transforms(original_image).unsqueeze(0).to(device)
                orig_size_tensor = torch.tensor([[orig_w, orig_h]], dtype=torch.float32).to(device)
                
                # Run inference
                results = model(model_input, orig_size_tensor)
                
                # Extract results
                if isinstance(results, (list, tuple)) and len(results) == 3:
                    # Format: (labels, boxes, scores)
                    labels, boxes, scores = results
                    labels = labels[0]  # First (and only) image in batch
                    boxes = boxes[0]
                    scores = scores[0]
                elif isinstance(results, list) and len(results) > 0:
                    # Format: list of dicts
                    result = results[0]
                    labels = result['labels']
                    boxes = result['boxes']
                    scores = result['scores']
                else:
                    print(f"Unexpected result format for image {image_id}")
                    continue
                
                # Count predictions above threshold
                above_thresh = (scores > args.threshold).sum().item()
                total_predictions += above_thresh
                
                # Create output filename
                if isinstance(image_id, torch.Tensor):
                    image_id = image_id.item()
                output_filename = f"vis_{image_id:06d}.jpg"
                output_path = output_dir / output_filename
                
                # Draw predictions on the original image and save
                draw_predictions(
                    original_image, labels, boxes, scores,
                    class_names=class_names,
                    threshold=args.threshold,
                    save_path=str(output_path)
                )
                
                processed_count += 1
                
                # Print progress periodically
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} images, "
                          f"Average predictions per image: {total_predictions/processed_count:.2f}")
    
    print(f"\nVisualization complete!")
    print(f"Processed {processed_count} images")
    print(f"Total predictions above threshold ({args.threshold}): {total_predictions}")
    print(f"Average predictions per image: {total_predictions/processed_count:.2f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main() 