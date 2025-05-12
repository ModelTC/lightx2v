# LightX2V: Light Video Generation Inference Framework

<div align="center">
  <picture>
    <img alt="LightLLM" src="assets/img_lightx2v.jpg" width=75%>
  </picture>
</div>

--------------------------------------------------------------------------------

## Supported Model List

✅ [HunyuanVideo-T2V](https://huggingface.co/tencent/HunyuanVideo)

✅ [HunyuanVideo-I2V](https://huggingface.co/tencent/HunyuanVideo-I2V)

✅ [Wan2.1-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

✅ [Wan2.1-I2V](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)

✅ [Wan2.1-T2V-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)

✅ [SkyReels-V2-DF](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P)


## How to Run

Please refer to the [documentation](https://github.com/ModelTC/lightx2v/tree/main/docs) in lightx2v

## Contributing Guidelines

We have prepared a `pre-commit` hook to enforce consistent code formatting across the project. If your code complies with the standards, you should not see any errors, you can clean up your code following the steps below:

1. Install the required dependencies:

```shell
pip install ruff pre-commit
```

2. Then, run the following command before commit:

```shell
pre-commit run --all-files
```

3. Finally, please double-check your code to ensure it complies with the following additional specifications as much as possible:
  - Avoid hard-coding local paths: Make sure your submissions do not include hard-coded local paths, as these paths are specific to individual development environments and can cause compatibility issues. Use relative paths or configuration files instead.
  - Clear error handling: Implement clear error-handling mechanisms in your code so that error messages can accurately indicate the location of the problem, possible causes, and suggested solutions, facilitating quick debugging.
  - Detailed comments and documentation: Add comments to complex code sections and provide comprehensive documentation to explain the functionality of the code, input-output requirements, and potential error scenarios.

Thank you for your contributions!
