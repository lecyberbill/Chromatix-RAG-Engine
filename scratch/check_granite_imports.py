try:
    from docling.datamodel import vlm_model_specs
    from docling.pipeline.vlm_pipeline import VlmPipeline
    print("SUCCESS: Imports for Granite Docling are working.")
    print(f"Spec available: {vlm_model_specs.GRANITEDOCLING_TRANSFORMERS}")
except Exception as e:
    print(f"ERROR: {e}")
