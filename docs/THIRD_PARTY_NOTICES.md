# 第三方原生依赖与许可来源

核对日期：2026-09-07。本文件记录项目中的 NCNN、Qualcomm QNN 依赖及对应上游材料，不替代依赖各自的完整许可条款。

## NCNN

- 上游：[Tencent/ncnn](https://github.com/Tencent/ncnn)，本地 Android 开发包版本为 `1.0.20260526`，来源包名为 `ncnn-20260526-android-vulkan`。
- 项目使用 `app/src/main/cpp/third_party/ncnn/include/ncnn/` 中的头文件，以及 `app/src/main/jniLibs/arm64-v8a/libncnn.so`。
- 许可：[项目附带的原文](../app/src/main/cpp/third_party/ncnn/LICENSE)，从本地 `ncnn-source.zip` 根目录的 `LICENSE.txt` 原样复制，包含 BSD 3-Clause 和该文件列出的第三方声明。可对照[上游 20260526 版本许可](https://github.com/Tencent/ncnn/blob/20260526/LICENSE.txt)。
- 许可文件 SHA-256：`7c974bac98848df46be1af5bdaa3c3c9c01f6082a90f55caeb7f60c6208aa255`。

来源核对：项目的 29 个头文件中，28 个与本地 Android 开发包逐字节一致；`ncnn_export.h` 的差别是共享库导出/隐藏符号的可见性宏。源码 ZIP 中可以直接对应的 25 个头文件全部与项目一致。本地 Android 开发包提供的是静态 `.a` 库，因此上述核对不表示项目中的 `libncnn.so` 是官方发布的预编译共享库。

NCNN 允许按其 BSD 3-Clause 条款分发源码和二进制；分发时应保留版权、条件及免责声明。启用 Vulkan 时还需保留实际链接组件的声明。本地 NCNN 源码包的 `glslang/LICENSE.txt` 已原样复制为 [glslang 完整许可](../app/src/main/cpp/third_party/ncnn/LICENSES/glslang-LICENSE.txt)，SHA-256 为 `17e70c676e1521ff3e4686f04a2053d93a7e28a33be8de7ec37ab0ff72feb677`。该文件保留 glslang 自身列出的多项许可，不能用 NCNN 自身的 BSD 声明替代。

来源范围：这些许可文件来自与项目头文件匹配的本地 NCNN 源码包；本次核对没有取得 `libncnn.so` 的完整构建记录，因而没有进一步证明二进制中 glslang 的精确提交号。发布新的原生库时，应同时保留其实际源码版本、构建记录和对应许可。

## Qualcomm QNN / QAIRT

构建与 HTP 后端使用 Qualcomm QAIRT SDK。已核对的本地版本为 `2.50.0.260828`，原始包为 `v2.50.0.260828.zip`。公共 Git 仓库不包含 QNN SDK 头文件或 `libQnn*.so`；构建时从外部 SDK 获取这些文件。SDK 应通过 Qualcomm 官方渠道获取，并按照下载时的协议使用。QNN 通用头文件包含 Qualcomm 专有声明，不应将它们视作 NCNN BSD 许可覆盖的文件。

已逐个核对本地项目的以下运行库：其 SHA-256 全部与该 QAIRT ZIP 中的对应文件一致。

| 运行库 | SDK 内相对目录 |
| --- | --- |
| `libQnnHtp.so` | `lib/aarch64-android/` |
| `libQnnSystem.so` | `lib/aarch64-android/` |
| `libQnnHtpV75Stub.so` | `lib/aarch64-android/` |
| `libQnnHtpV75CalculatorStub.so` | `lib/aarch64-android/` |
| `libQnnHtpV75.so` | `lib/hexagon-v75/unsigned/` |
| `libQnnHtpV75Skel.so` | `lib/hexagon-v75/unsigned/` |
| `libQnnHtpV73Stub.so` | `lib/aarch64-android/` |
| `libQnnHtpV73CalculatorStub.so` | `lib/aarch64-android/` |
| `libQnnHtpV73.so` | `lib/hexagon-v73/unsigned/` |
| `libQnnHtpV73Skel.so` | `lib/hexagon-v73/unsigned/` |
| `libQnnHtpV69Stub.so` | `lib/aarch64-android/` |
| `libQnnHtpV69CalculatorStub.so` | `lib/aarch64-android/` |
| `libQnnHtpV69.so` | `lib/hexagon-v69/unsigned/` |
| `libQnnHtpV69Skel.so` | `lib/hexagon-v69/unsigned/` |

v73 的四份是 2026-11-15 为骁龙 8 Gen 2 / 8s Gen 3 的 HTP 后端新增的；它们与同一次核对中的 ZIP
条目逐字节一致（SHA-256 分别为 `d4749b59…bcc0a`、`876c5548…5c79`、`bab126f2…85e7de`、
`e3070ed8…fb2dd`）。v69 的四份同日为骁龙 8 Gen 1 / 8+ Gen 1 新增，同样逐字节核对过
（SHA-256：`libQnnHtpV69Stub.so` = `b7063e68…46d6`、`libQnnHtpV69CalculatorStub.so` = `b773196c…a05c`、
`libQnnHtpV69.so` = `1d1bcfec…77e6`、`libQnnHtpV69Skel.so` = `93c395d8…9e55`）。
这些文件同样不进入公共仓库（`.gitignore` 覆盖），构建时从外部 SDK 获取。

上述文件对应 QAIRT 2.50，不能用本地缓存中 `com.qualcomm.qti:qnn-runtime:2.42.0` 的版本号作为来源说明。

原始包中 `qairt/2.50.0.260828/` 目录包含以下完整许可材料：

- `LICENSE.pdf`：标题为 `AI Stack License`，共 3 页。SHA-256 为 `ec1dccfdcba5c6e64126e84199b8362bf4999107bfa567ebe831dbb4c461692b`。
- `NOTICE.txt` 和 `QNN_NOTICE.txt`：两个文件逐字节相同，SHA-256 为 `0c5e8aad3506d0ab881cabaf0dae8de64e1784dc1ee6c873caf24a78fbf01924`。
- `QNN_TFLITE_DELEGATE_NOTICE.txt`、`NOTICE_WINDOWS.txt`：分别面向对应组件/平台，不能当作 Android QNN 原生运行库声明的替代文件。

`LICENSE.pdf` 第 1 页第 1 条允许按条款将对象代码集成在自己的应用中分发，同时明确没有授予独立分发该软件的许可。第 2(b) 条要求保留专有声明，第 10(h) 条要求保留第三方 Notice。因此，随应用集成分发的条款不能直接作为把 SDK 头文件或独立 `.so` 上传到公共源码仓库的授权依据；公共源码发布应让构建者自行取得 SDK。

发布包含 QNN 的 APK 时，应将外部 SDK 根目录的原始 `LICENSE.pdf` 和 `NOTICE.txt` 一同复制到 APK 的第三方声明资源中；只复制 SDK 的 `include/`、`lib/` 目录会漏掉这些文件。本文是来源记录，不能代替原始包中的 `LICENSE.pdf` 和 `NOTICE.txt`。
