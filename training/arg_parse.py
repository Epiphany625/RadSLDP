import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune VLM models with optional LoRA and Differential Privacy")
    
    # Model and data arguments
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_id", type=str, required=True, help="Model ID from HuggingFace")
    parser.add_argument("--train_image_folder", type=str, required=True, help="Path to training images")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to training dataset JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saved model")
    parser.add_argument("--no_noise", action="store_true", default=False,
                    help="Clipping only, no Gaussian noise added (ablation)")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (must be 1 for dp)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")

    # Differential Privacy arguments
    parser.add_argument("--use_dp", action="store_true", help="Enable differential privacy training")
    parser.add_argument("--target_epsilon", type=float, default=8.0, help="Target epsilon for DP (privacy budget)")
    parser.add_argument("--target_delta", type=float, default=1e-5, help="Target delta for DP")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for DP clipping")
    
    # LDP 
    parser.add_argument("--use_ldp", action="store_true", help="Use Local DP on sensitive token embeddings")
    parser.add_argument("--clip_norm", type=float, default=1, help="clip norm")
    parser.add_argument("--selective_ldp", action="store_true", help="Use selective LDP (only sensitive tokens)")


    return parser.parse_args()

def parse_prediction_args():
    parser = argparse.ArgumentParser(description="Validation / evaluation arguments")
    parser.add_argument("--model_id", type=str, required=True, help="Model base. If full finetuned, please input a random value.")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="The directory where the finetuned model lives.")
    parser.add_argument("--test_image_folder", type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to the test dataset file (e.g., JSON/CSV)")
    parser.add_argument("--prediction_file", type=str, required=True, help="Directory to save evaluation outputs")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA evaluation path")
    return parser.parse_args()

def parse_evaluation_args():
    parser = argparse.ArgumentParser(description="Evaluate MIMIC-CXR model predictions")

    parser.add_argument("--results_file", type=str, required=True, help="Path to JSONL file with predictions and ground truths")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--scorers", nargs="+", default=["CheXbert", "F1-RadGraph", "BLEU-1", "BLEU-4", "ROUGE-L"], help="List of scorers to use")
    parser.add_argument("--bootstrap_ci", action="store_true", help="Compute bootstrap confidence intervals")
    parser.add_argument("--no_bootstrap_ci", dest="bootstrap_ci", action="store_false", help="Do not compute bootstrap confidence intervals (faster)")

    parser.set_defaults(bootstrap_ci=True)
    return parser.parse_args()