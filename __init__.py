from .nodes import UnloadModelNode, UnloadAllModelsNode

NODE_CLASS_MAPPINGS = {
    "UnloadModel": UnloadModelNode,
    "UnloadAllModels": UnloadAllModelsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnloadModel": "Unload Model",
    "UnloadAllModels": "Unload All Models",
}

__version__ = "1.0.0"