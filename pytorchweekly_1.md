# Pytorch Weekly - #1, May 2nd 2023

Welcome to the 1st issue of Pytorch Weekly, a weekly newsletter covering development in Pytorch AI development platform. You can subscribe the newsletter with [pytorchweekly@freelists.org](https://www.freelists.org/list/pytorchweekly) or [Lingcc/pytorchweekly (github.com)](https://github.com/Lingcc/pytorchweekly) .

## News and articles from around the web and events

- [Hidet](https://pytorch.org/blog/introducing-hidet/) is introduced on `PyTorch` blog as a deep learning compiler for Efficient Model serving.  `Triton` and `Hidet Script` both allow tensor program developers can easily handle the tile-based programming model.While, compared to `Triton`, `Hidet Script` simplifies tensor programming by handling the fine-grained computation and memory resources (e.g., warps, shared memory) manipulation.

- [TorchBench](https://arxiv.org/abs/2304.14226) is introduced by `Yueming Hao` and other guys from `Meta Platforms, Inc`. `TorchBench` is a novel benchmark suite to study the performance of `PyTorch` software stack and has been used to identify the GPU performance inefficiencies in PyTorch, and it has also been integrated into the PyTorch continuous integration system.

- Towards Data Science published an amazing article: [Build your own Transformer from scratch using Pytorch](https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb) writen by Arjun Sarkar. It teaches the reader to build a transformer model step by step in PyTorch.

- The latest `PyTorch 2.0 Ask the Engineers Q&A Series` brought `TorchRL` by [Vincent Moens](https://github.com/vmoens/) and [Shashank Prasanna](https://shashankprasanna.com/) from `Meta`.

- [Zachary DeVito](https://cs.stanford.edu/~zdevito/) contribute to the Pytorch Forum about  [Fast combined C++/Python/TorchScript/Inductor tracebacks](https://dev-discuss.pytorch.org/t/fast-combined-c-python-torchscript-inductor-tracebacks/1158/2)

- [David Stutz](https://github.com/davidstutz) proposed a way for [Loading and Saving PyTorch Models Without Knowing the Architecture in Advance](https://davidstutz.de/loading-and-saving-pytorch-models-without-knowing-the-architecture/)

- Want to check the differences between `PyTorch` and `Jax`? check [JAX vs. PyTorch: Differences and Similarities [2023]](https://geekflare.com/jax-vs-pytorch/)

## On the forums and maillists

- [Run PyTorch on Multiple GPUs](https://discuss.pytorch.org/t/run-pytorch-on-multiple-gpus/20932/75) thread was actived again since `SM2023` tried to fine tune the GPT-2 model on multiple GPUs. Run model on multiple GPUs are not easy to handle, especially for load balance and parallel optimizations. Fresh guys are always recommanded to go through the [Multi-GPU examples tutorials](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html). Thanks to `ptrblck`

- According to [Would pytorch for cuda 11.6 work when cuda is actually 12.0](https://discuss.pytorch.org/t/would-pytorch-for-cuda-11-6-work-when-cuda-is-actually-12-0/169569/5), PyTorch binary currently shipped directly with `CUDA`，`CUDNN`, and `cuBLAS`, etc, it uses `11.7` and `11.8` by default. And only when build PyTorch from source, will it use the loca installed CUDA toolkit. You are recommanded to use [the install method](https://pytorch.org/get-started/locally/)

- Result reproducibility is always a headache for ML training. The thread [Different training results on different machines](https://discuss.pytorch.org/t/different-training-results-on-different-machines-with-simplified-test-code/59378/22) has lasted for more than 2 years disscussing about this. PyTorch doc [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) has also mentioned that Pytorch does not guarante completely reproducible results. The thread added a new difference between Windows and Linux which might cause unproducable result since `os.listdir` or `glob.glob` on Windows produce an ordered list by default, however Linux output the random file list. it lead to different result.

- [JOROZCO](https://discuss.pytorch.org/u/jorozco/summary) proposed a way to [convert PyTorch model to `ONNX` format](https://discuss.pytorch.org/t/convert-pytorch-model-to-onnx-format-inference-not-same/145005/8)

- [How to fix “CUDA error: device-side assert triggered” error?](https://discuss.pytorch.org/t/how-to-fix-cuda-error-device-side-assert-triggered-error/137553/5) introduced `CUDA_LAUNCH_BLOCKING=1` to disable asynchronous kernel launhches.  

## PyTorch commits Highlight

- PyTorch main develop branch changed from `master` to `main`

- [CUDA 12.1 build is enabled again on windows](https://github.com/erlv/hpytorch_nocuda/commit/0a5c9304997631ba84f7185a1c7e29d0b926974a)

- Plenty of `Dynamo`, `Triton` bug fixes and improvement, such as [add support for serializing real tensor data in after AOT minifier](https://github.com/pytorch/pytorch/pull/99834), [Basic dynamo support for traceable collectives](https://github.com/pytorch/pytorch/pull/94440), [Introduce FXGraphExtractor into torch.onnx.dynamo_export](https://github.com/pytorch/pytorch/pull/99940)

- [Dan Dale fix a CPU offload performance issue for `ShardedGradScaler`](https://github.com/pytorch/pytorch/pull/100108). The performance analyze of the work is amazing.

- Related changes to remove CUDA 11.6 support

- [Improve the debug method for after AOT accuracy debugging](https://github.com/pytorch/pytorch/pull/100226)

- Improve New Architecture support: [Making FSDP device-agnositc for custom-backend which implement cuda-semantics](https://github.com/pytorch/pytorch/pull/99024), [New hook for MTIA architecture](https://github.com/pytorch/pytorch/pull/99854)

- [Optimized EMA implementation](https://github.com/pytorch/pytorch/pull/94820)

- [Update Cutlass to v3.1](https://github.com/erlv/hpytorch_nocuda/commit/dfba65be8b1b42175905e609f9eb5db431c449e0)

## Other project or company weekly updates highlight

- Modular AI annouced its two init products. The first one is [the fastest unified AI inference engine in the world](https://www.modular.com/engine). The second one is a [new programming language for all AI developers `Mojo`](https://www.modular.com/mojo)