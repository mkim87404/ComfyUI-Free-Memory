import comfy.model_management as model_management
import gc
import torch

# AnyType for passthrough (required for ComfyUI graph compatibility, may not work with reroutes)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class UnloadModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "persist_any_1": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata. At least 1 passthrough is required to anchor this node's execution at an intended point in the workflow.", }),
                "synchronize_cuda": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to explicitly torch.cuda.synchronize() before clearing all cache. Recommended but not necessary. Ignored for Non-CUDA users."
                }),
            },
            "optional": {
                "persist_any_2": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_3": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_4": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_5": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_6": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_7": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "model": (any, { "tooltip": "model to unload (e.g. Diffusion, UNet, LoRAs, VAE, CLIP, LLM, etc.) All other models will remain untouched.", }),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
    
    RETURN_TYPES = (any, any, any, any, any, any, any)
    RETURN_NAMES = ("persist_any_1", "persist_any_2", "persist_any_3", "persist_any_4", "persist_any_5", "persist_any_6", "persist_any_7")
    OUTPUT_TOOLTIPS = ("Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.",) * 7
    FUNCTION = "route"
    CATEGORY = "Free Memory"
    DESCRIPTION = "Unload a model and release its associated VRAM/RAM usage at any point in the workflow. Optionally persist/route any data through to the next node."
    
    def route(self, synchronize_cuda: bool, model=None, persist_any_1=None, persist_any_2=None, persist_any_3=None, persist_any_4=None, persist_any_5=None, persist_any_6=None, persist_any_7=None):
        print("### UnloadModelNode • Model VRAM/RAM cleanup ###")
        
        try:
            loaded_models = model_management.loaded_models()    # returns a list of the actual model objects currently tracked in current_loaded_models. This is not just Python refs — it's the official list ComfyUI uses for memory decisions.
            if model is not None:
                if model in loaded_models:
                    print(" - Model found in memory, unloading...")
                    loaded_models.remove(model) # this is to let the later model_management.free_memory(,,loaded_models) skip the unloading of every model other than "model"
                elif isinstance(model, dict) and 'model' in model:
                    print(f" - Deleting model reference of type {type(model['model']).__name__}")
                    del model['model']  # Manually delete the model rather than delegating to model_management.free_memory()
                    # Emptying the cache after this should free the memory.

            model_management.free_memory(1e30, model_management.get_torch_device(), loaded_models)
            # This is the core "free as much VRAM as possible" function, where "1e30" means "free this absurdly large amount of memory (i.e. everything)". It walks current_loaded_models, skips unloading anything in keep_loaded (third argument), calls model_unload() on the rest (which does the model unload/detach, unpatch weights, set real_model=None, and for some models a controlled partially_unload() to the offload device - no forced CPU offload unless the model itself decides to partially unload), pops them from the internal list "current_loaded_models", then calls cleanup_models_gc() (which conditionally runs gc.collect() + soft_empty_cache() if any momdel in current_loaded_models is_dead() i.e. memory leak is suspected) and soft_empty_cache() once (if it unloaded at least 1 model).
            # This is the official ComfyUI maintained method that safely manages its internal model load states through current_loaded_models entries and other internal memory accounting, so I have to accept its one internal call to torch.cuda.synchronize() regardless of my "synchronize_cuda" node config setting.
            print(" - Model unloaded")

            if synchronize_cuda:
                print(" - Synchronizing hardware accelerator")
                model_management.soft_empty_cache(True)
                # This is a Device-agnostic wrapper that does:
                # CUDA → torch.cuda.synchronize() + torch.cuda.empty_cache() + torch.cuda.ipc_collect()
                # MPS / XPU / NPU / MLU → runs the equivalent empty_cache for that backend, and the force param is ignored in current ComfyUI (legacy).
                # DESIGN: While this is a convenient one-liner for cross-device safety, I've disabled it by default and extracted out the cross-platform support logic + empty_cache + ipc_collect below, because calling torch.cuda.synchronize() blocks execution until all queued CUDA work finishes, which is generally not necessary.
        except Exception as e:
            print(f" - Non-fatal error during unload: {e}")
        finally:
            # First pass (Device-aware empty_cache)
            print(" - Clearing VRAM cache")
            if torch.cuda.is_available():   # NVIDIA
                torch.cuda.empty_cache()    # releases the cached VRAM and available memory held by the allocator but not currently in use back to the system.
            elif hasattr(torch, 'mps') and torch.mps.is_available():    # Apple Silicon
                torch.mps.empty_cache()
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():    # Intel XPU
                torch.xpu.empty_cache()
            elif hasattr(torch, 'npu') and torch.npu.is_available():    # Ascend NPU
                torch.npu.empty_cache()
            elif hasattr(torch, 'mlu') and torch.mlu.is_available():    # Cambricon MLU
                torch.mlu.empty_cache()

            print(" - Clearing RAM")
            gc.collect()    # triggers Python’s gc.collect() to release objects from memory that no longer have active references. Critical for freeing CPU RAM + any Python object references after the tensors are gone.

            # Second pass (catches anything GC just released) + IPC only on CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()    # double empty_cache is a well-known best practice to catch lingering tensors after GC
                torch.cuda.ipc_collect()    # frees any lingering CUDA IPC (inter-process communication) / shared-memory handles that empty_cache() sometimes misses. Useful when models were loaded with certain GGUF/quantized loaders or in multi-process scenarios. Harmless in a normal single-process ComfyUI server. Calling it once after the 2nd empty_cache() call is sufficient, because empty_cache() and gc.collect() do not create, modify, or depend on IPC handles, and ipc_collect() is idempotent: once the IPC handles are collected, calling it again does nothing and the driver returns immediately.
                model_management.cleanup_models()
                # This is a lightweight "remove dead/stale model wrappers" helper that scans current_loaded_models and removes entries where real_model() is None (i.e. dead wrappers that free_memory may have left behind in some edge cases), pops them from the global current_loaded_models list, and deletes the wrapper.
                # which is a harmless & cheap extra safety pass (no cache clearing, no soft_empty_cache(), no synchronize(), no model_unload() call, no GC) especially after the explicit del + gc.collect() + double empty_cache() path in the single-model node that frees the Python tensors/references you control, but does not automatically clean stale wrappers from ComfyUI’s global current_loaded_models list if the model object becomes unreachable or real_model() becomes None.
                # only meaningful after the model tensors have already been fully unloaded + garbage-collected (i.e. after the free_memory/unload_all_models + gc.collect() + empty_cache() have done their job) and hence every last dead model wrapper whose real_model() just became None can be caught.
                print(" - Clearing CUDA stats")
                try:
                    torch.cuda.reset_peak_memory_stats()    # Optional stats reset once at the very end. try catch because this can raise in edge cases (no active CUDA context, older PyTorch, or after certain errors)
                except:
                    pass
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()
                model_management.cleanup_models()
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
                model_management.cleanup_models()
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.empty_cache()
                model_management.cleanup_models()
            elif hasattr(torch, 'mlu') and torch.mlu.is_available():
                torch.mlu.empty_cache()
                model_management.cleanup_models()
            
            print(" - VRAM/RAM cleanup complete.")

        return (persist_any_1, persist_any_2, persist_any_3, persist_any_4, persist_any_5, persist_any_6, persist_any_7)

class UnloadAllModelsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "persist_any_1": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata. At least 1 passthrough is required to anchor this node's execution at an intended point in the workflow.", }),
                "synchronize_cuda": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to explicitly torch.cuda.synchronize() before clearing all cache. Recommended but not necessary. Ignored for Non-CUDA users."
                }),
            },
            "optional": {
                "persist_any_2": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_3": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_4": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_5": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_6": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
                "persist_any_7": (any, { "tooltip": "Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.", }),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = (any, any, any, any, any, any, any)
    RETURN_NAMES = ("persist_any_1", "persist_any_2", "persist_any_3", "persist_any_4", "persist_any_5", "persist_any_6", "persist_any_7")
    OUTPUT_TOOLTIPS = ("Persist any data throughout the memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.",) * 7
    FUNCTION = "route"
    CATEGORY = "Free Memory"
    DESCRIPTION = "Unload all models and release all associated VRAM/RAM usage at any point in the workflow. Optionally persist/route any data through to the next node."

    def route(self, synchronize_cuda: bool, persist_any_1=None, persist_any_2=None, persist_any_3=None, persist_any_4=None, persist_any_5=None, persist_any_6=None, persist_any_7=None):
        print("### UnloadAllModelsNode • Full VRAM/RAM cleanup ###")

        try:
            model_management.unload_all_models()    # This just calls free_memory(1e30, get_torch_device()) with no keep_loaded models list provided defaulting it to [] (i.e. unload everything).
            print(" - All models unloaded")

            if synchronize_cuda:
                print(" - Synchronizing hardware accelerator")
                model_management.soft_empty_cache(True)
        except Exception as e:
            print(f" - Non-fatal error during unload: {e}")
        finally:
            print(" - Clearing VRAM cache")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.empty_cache()
            elif hasattr(torch, 'mlu') and torch.mlu.is_available():
                torch.mlu.empty_cache()

            print(" - Clearing RAM")
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                model_management.cleanup_models()
                print(" - Clearing CUDA stats")
                try:
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()
                model_management.cleanup_models()
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
                model_management.cleanup_models()
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.empty_cache()
                model_management.cleanup_models()
            elif hasattr(torch, 'mlu') and torch.mlu.is_available():
                torch.mlu.empty_cache()
                model_management.cleanup_models()

            print(" - VRAM/RAM cleanup complete.")

        return (persist_any_1, persist_any_2, persist_any_3, persist_any_4, persist_any_5, persist_any_6, persist_any_7)