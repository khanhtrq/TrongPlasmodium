from TrongPlasmodium.Khanh_inference_simple_draft import run_simple_inference

def main():
    """Main function for standalone script execution."""
    print("🎯 PlasmodiumClassification Simple Inference")
    print("=" * 50)
    model_checkpoint = "./model/final_bbbc041.pth"
    model_name = 'efficientnet_b1.ra4_e3600_r240_in1k'
    model_num_classes = 6

    crop_image_folder = "draft_data"
    
    # Example usage - specify direct model name, checkpoint path, and class count
    results = run_simple_inference(
        model_name=model_name,  # Direct timm model name
        model_checkpoint=model_checkpoint,  # Direct path
        model_num_classes=model_num_classes,  # Explicitly specify model class count
        split='test',
        batch_size=16,
        config_path = 'config_draft.yaml',
        save_scores=True,  # 💾 Enable softmax score saving
        scores_filename="test_scores_6cls_vs_7cls.txt",  # Custom filename
        run_phase2=True,  # 🔬 Enable Phase 2 evaluation
        verbose=True,
        imgf_root = crop_image_folder
    )

    print("RESUTLS IN MAIN:\n-------------------")
    print(results['inference_results']['predictions'])
    print(results['inference_results']['confidences'])


if __name__ == '__main__':
    main()