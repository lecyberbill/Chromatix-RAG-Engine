from docling.datamodel import vlm_model_specs
from docling.datamodel.pipeline_options import VlmPipelineOptions, AcceleratorDevice

vlm_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
)

print(f"VlmPipelineOptions fields: {vlm_options.__dict__.keys()}")
print(f"InlineVlmOptions fields: {vlm_options.vlm_options.__dict__.keys()}")

try:
    vlm_options.device = AcceleratorDevice.CPU
    print("SUCCESS: vlm_options.device works.")
except Exception as e:
    print(f"ERROR on vlm_options.device: {e}")

try:
    vlm_options.vlm_options.device = AcceleratorDevice.CPU
    print("SUCCESS: vlm_options.vlm_options.device works.")
except Exception as e:
    print(f"ERROR on vlm_options.vlm_options.device: {e}")
