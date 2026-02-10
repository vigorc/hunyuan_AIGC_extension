# 混元 AIGC 扩展 (Hunyuan AIGC Extension)

Isaac Sim 扩展：通过**单张照片**或**文本 Prompt** 生成带物理属性的 USD 资产，支持腾讯云 API 与本地 Hunyuan3D 模型调用。

---

## 📖 概述 (Overview)

本扩展为 **NVIDIA Isaac Sim / Omniverse** 提供 AIGC 3D 生成能力，支持从单张图片或文本描述生成带物理属性的 USD 资产，便于在仿真中直接使用。支持**腾讯云 API**与**本地部署的 Hunyuan3D 模型**两种调用方式。

This extension provides AIGC 3D generation for **NVIDIA Isaac Sim / Omniverse**: generate **physics-ready USD assets** from a **single image** or **text prompt**, with support for **Tencent Cloud API** and **local Hunyuan3D model** inference.

## ✨ 功能特性 (Features)

- **图像上传 / Image Upload**: 上传单张图像或整个文件夹
- **文本生成 / Text-to-3D**: 通过 Prompt 生成 3D 模型
- **3D 生成 / 3D Generation**: 基于混元（Hunyuan3D）从图像或文本生成 3D 模型
- **物理支持 / Physics Support**:
  - 可变形物理（Deformable），可自定义参数
  - 刚体物理（Rigid Body），多种碰撞体类型
- **自动加载 / Auto-loading**: 自动将生成的模型加载到 Stage
- **缩放工具 / Scale Tool**: 将生成模型缩放到真实世界尺寸

## 🔌 调用方式 (API Modes)

### 1. 腾讯云 API (Tencent Cloud API)

在扩展配置中填写腾讯云 3D 生成 API 的 Key 与 URL（如 `config.json` 中的 `cloud_api_key`、`cloud_api_url`），即可通过云端服务生成 3D 资产，无需本地 GPU。

### 2. 本地模型 (Local Model)

使用 [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) 本地部署 API 服务后，将扩展中的**服务器 URL** 指向本地地址（默认 `http://localhost:8081`）即可。

**本地 API 启动参考**：  
在 [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) 仓库中运行：

```bash
# 参考官方仓库中的 api_server.py 启动本地服务
python api_server.py
```

具体接口与参数请见官方文档：[Hunyuan3D-2.1 - api_server.py](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)

## 🚀 使用方法 (Usage)

### 3D 生成 (3D Generation)

1. 使用 **Select Image** 或 **Select Folder** 选择图像，或切换到文本模式输入 **Prompt**
2. 在扩展中选择调用方式：**腾讯云 API** 或 **本地服务器**，并配置对应 URL / API Key
3. 点击 **Generate 3D Model** 开始生成
4. 生成的模型将保存并可选择自动加载到当前 Stage

### 缩放工具 (Scale Tool)

1. 在视口中选择一个 Prim（模型）
2. 点击 **Get Selected Prim** 查看当前尺寸
3. 输入物体实际高度（米）
4. 选择高度轴（Y / X / Z，默认 Y）
5. 点击 **Scale to Height** 应用统一缩放

**示例**：椅子生成高度为 2.0m，实际椅子为 0.95m → 输入目标高度 0.95m，即可缩放到真实比例。

## 📋 要求 (Requirements)

- **NVIDIA Isaac Sim / Omniverse**
- 3D 生成任选其一：
  - **腾讯云 API**：有效 API Key 与可访问的云端 URL
  - **本地模型**：本地运行 [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) 的 `api_server.py`（如默认 `http://localhost:8081`）

## 📄 许可证 (License)

详见 `PACKAGE-LICENSES` 目录。

---

**Made with ❤️ for NVIDIA Isaac Sim / Omniverse**
