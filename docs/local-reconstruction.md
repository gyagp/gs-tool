# 从视频获得更逼真的模型（Windows 本机处理）

网页中的视频导入仍是手机可运行的快速轮廓预览。需要保留标签、瓶盖纹路、不规则形状和不同方向的反光时，使用这里的 **真实机位恢复 + 多视角高斯训练**，完成后把 PLY 导入 [网页查看器](https://gs.gyagp.workers.dev)。这条流程不使用圆瓶旋转体或假定均匀环拍机位，也不上传视频。

## 一次准备

需要 Windows x64、Python 3.11–3.13，以及支持 Vulkan/WebGPU 的显卡和最新驱动。已在 Intel Arc 集成显卡上实际跑通，不要求 NVIDIA/CUDA。处理速度和可用分辨率取决于显存。首次安装需要网络下载开源工具；训练期间媒体保留在本地。

在项目目录运行 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-reconstruction.ps1
```

安装使用独立 Python 环境，不更改系统 Python 依赖。Brush v0.3.0 从官方 release 下载并核对 SHA-256。COLMAP/Python 依赖固定版本，来源和许可见 [第三方声明](../THIRD_PARTY_NOTICES.md)。

## 重建自己的视频

```powershell
artifacts/tools/reconstruction-python/Scripts/python.exe -u scripts/reconstruct_video.py `
  'C:\path\to\orbit.mp4' --output artifacts/my-scan
```

默认从 360 个时间候选中保留最多 120 个清晰、不同的帧，恢复真实内参与机位，提取主体，联合优化 16,000 步并保留三阶球谐颜色。最多 400,000 个高斯是资源上限，不是强行补足的数量。每八帧留出一帧作验证；这些图像的颜色/透明度不会参与高斯训练。相机定位使用所有帧，因此这是已标定的留出视角验证，不能视为完全独立的新拍摄测试。

120 帧以内的小环拍序列默认进行全连接匹配以闭合环路（160 帧以下均启用）。更多帧可用 `--frames 240 --exhaustive`，代价是显著更长的 CPU 匹配时间。增加帧数只有在帧中包含新的角度、清晰度或遮挡信息时才有帮助。

默认保留至 1920 像素长边，低于该分辨率的源帧不会放大。显存充足时用 `--max-dimension 0` 保留原始分辨率，先以较小任务验证内存。输出目录对应一段视频及其抽帧设置；更换视频、帧数或分辨率时使用新目录。

进度见 `artifacts/my-scan/progress.json`；GPU 训练日志见 `trained/train.log`。机位覆盖不足、主体分割失败时会报错，不用猜测机位或替代几何掩盖问题。可用 `--stages segment,prepare,train,evaluate` 从指定阶段继续；不要在同一训练目录覆盖已有模型，比较新训练用 `--training-name experiment-2`。

## 查看结果和比较

1. 打开 `artifacts/my-scan/trained/validation/held-out-comparison.jpg`：左列是留出原帧，中间是早期模型，右列是最终模型。检查正面、侧面、背面的文字、轮廓及反光。
2. 打开 `validation/report.json`：以瓶子区域的 PSNR/SSIM 为主要参考。全图指标包含大面积背景，通常偏高。指标不能代替转动检查。
3. 把 `validation/best.ply` 导入网页。它按留出帧的主体区域 SSIM 选择已有检查点；选择依据记录在 `best.json`。检查点仍保存在 `trained/`。
4. 网页的原始高斯模式保留完整点数、椭球形状和三阶球谐。诊断模式仅显示基础颜色。导出 PLY 返回原始文件，避免再次丢失精度；导入模型后禁用会丢失球谐的 SPLAT 转换。

单独重新验证已有训练：

```powershell
artifacts/tools/reconstruction-python/Scripts/python.exe scripts/evaluate_reconstruction.py artifacts/my-scan
```

也可在预览服务器运行时测试实际网页的导入/转动/导出；测试不上传模型：

```powershell
$env:GS_TEST_MODEL = 'D:\path\to\best.ply'
npm run test:model
```

使用准备阶段导出的真实机位验证网页颜色与细节（自动启动临时本机服务器）：

```powershell
$env:GS_RECONSTRUCTION = 'D:\path\to\my-scan'
npm run test:model-quality
artifacts/tools/reconstruction-python/Scripts/python.exe scripts/evaluate_reconstruction.py `
  $env:GS_RECONSTRUCTION --renders "$env:GS_RECONSTRUCTION/browser-validation"
```

该验证使用居中的等焦距针孔机位；未满足这种标定时会明确停止。查看器使用与标准/Brush 训练一致的高斯滤波和深度排序，在支持时用半精度浮点缓冲混合透明高斯，避免 8 位画布逐层混合的颜色损失。普通显示设备仍会最终量化到屏幕色深。没有浮点渲染扩展的设备使用 8 位混合，可能出现颜色误差。

## 适合什么拍摄

算法不限定瓶子形状，但需要物体和场景静止、足够的特征与相邻帧重叠。固定物体，移动手机绕拍；建议再拍一圈略高的角度，看到瓶盖、顶面和凹陷。保持对焦/曝光稳定，让主体占据较大画面，避免裁断顶部和底部。分割模型默认寻找主要的中心主体，复杂背景、多物体或细小透明零件需要检查 `masks/` 和 RGBA 训练图。

纯色、反光和透明区域仍可能漂浮或模糊；没拍到的底面、标签背后和凹陷无法可靠恢复。环拍过程光照/自动曝光变化也会限制颜色一致性。本流程已用一段真实饮料瓶视频验证，不代表所有物体都能达到相同效果。

所有源视频、抽帧、模型和截图都放在忽略的 `artifacts/` 中，不进入 Git 或部署。网页导入直接读取本地文件；从手机查看同一私有结果，可把 PLY 通过自己的文件传输方式移到手机后导入。
