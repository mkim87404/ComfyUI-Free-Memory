# ComfyUI-Free-Memory

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Secure & device-agnostic ComfyUI custom nodes for unloading a model or all models at any point in a workflow, followed by a robust set of VRAM & RAM clean up operations, using the memory management utilities already present in ComfyUI, and optionally persisting any given data throughout this memory cleanup process. Forked from https://github.com/SeanScripts/ComfyUI-Unload-Model and inspired by https://github.com/ShmuelRonen/ComfyUI-FreeMemory

Includes two nodes: `Unload Model` and `Unload All Models`. These are used as passthrough nodes, so that you can unload one or all models at a specific step in your workflow.

These nodes can be used to introduce deliberate memory management points throughout your workflow e.g. to unload model(s) and their memory usage at a point where they are no longer needed for further progression into subsequent nodes.

An example use case is to unload the CLIP model after getting the conditioning output for the prompt, in order to save VRAM before commencing KSampling, and this is especially relevant when using Flux. This could be useful if you don't have enough VRAM to load both the Flux diffusion model and the T5XXL text encoder at the same time, or don't want to keep them both persistently loaded. I find this useful for having spare VRAM to keep a local LLM loaded. Unloading models could also be useful at the end of a workflow, or when switching between different models, if you want to manage your memory manually.

**Please note:** these nodes are dependent on the ComfyUI implementation of its `comfy.model_management.py` module, and may stop working in future ComfyUI updates. In particular, there have been some recent changes in the GGUF loader nodes that could cause the GGUF models to not actually unload (though this might be fixed already).

## 🍃 Added Changes (Post-Fork)

1. Added 7 optional "passthrough" arguments that can be used to persist any data throughout this node's memory cleanup process e.g. latents, conditioning, images, masks, and other metadata.
    - While this passthrough of data may not be strictly necessary for data persistence (i.e. as long as those data variables are referenced by some later nodes elsewhere in the workflow), this explicit passthrough design can still help to make the routing and execution order of nodes more deterministic and predictable, as well as tidying up & linearizing the node connections visually as your workflow gets bigger.

2. Separated out torch.cuda.synchronize() from comfy.model_management.soft_empty_cache(True) and surfaced it as an optional toggle on the node, as waiting for all globally queued asynchronous CUDA tasks to finish is not strictly necessary before empty_cache().

3. Double empty_cache() - before & after py garbage collection to catch any lingering tensors after GC

4. Added comfy.model_management.cleanup_models() and torch.cuda.reset_peak_memory_stats() to keep the ComfyUI internal model load tracking robust & CUDA stats clean after each run.

5. Device-agnostic & official ComfyUI-managed memory management utilities for secure & future-proof model unload and memory clearance behavior.

## 🛠️ Installation

1. Clone this repo into your `ComfyUI/custom_nodes` folder:
```bash
git clone https://github.com/mkim87404/ComfyUI-Free-Memory.git
```
2. Restart the ComfyUI server → double click anywhere in the workflow and search for **"Unload Model"** or **"Unload All Models"**

## 🚀 Usage

Add the `Unload Model` or `Unload All Models` node at any point in your workflow to unload a model/all models at that step. Connect at least 1 value into "persist_any_1" to anchor the node's memory clearance operation to a deterministic point in the workflow, and optionally the model you want to unload into "model" (e.g. Diffusion, UNet, LoRAs, VAE, CLIP, LLM, etc.), and any data you want to persist throughout the memory cleanup into "persist_any_N" (e.g. latents, conditioning, images, masks, and other metadata), then route the output of the node to wherever you would have routed the original input "persist_any_N", respecting the same order of input & output N.

For example, if you want to unload the CLIP models to save VRAM while using Flux, add this node after the "ClipTextEncode" or "ClipTextEncodeFlux" node, feeding the clip output conditioning into "persist_any_1", and feeding the CLIP model into "model", then route the output "persist_any_1" to wherever you would have sent the conditioning, e.g. "FluxGuidance" or "BasicGuider".

You don't need to input any model for the `Unload All Models` node as it will unload everything, and for the `Unload Model` node, if you don't pass any input model, it will just clear all cache & clean up the stats without unloading any model.

## 📜 License

[**MIT License**](https://github.com/mkim87404/ComfyUI-Free-Memory/blob/main/LICENSE) – feel free to use in any personal or commercial project, fork, or open issues/PRs – contributions and feedback all welcome!

Based on original works by [SeanScripts](https://github.com/SeanScripts/ComfyUI-Unload-Model), [willblaschko](https://github.com/willblaschko/ComfyUI-Unload-Models), [ShmuelRonen](https://github.com/ShmuelRonen/ComfyUI-FreeMemory).