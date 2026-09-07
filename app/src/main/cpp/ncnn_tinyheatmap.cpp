#include <jni.h>

#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

#include <ncnn/gpu.h>
#include <ncnn/layer_type.h>
#include <ncnn/mat.h>
#include <ncnn/net.h>

namespace {

constexpr int kInputWidth = 152;
constexpr int kInputHeight = 1024;
constexpr int kInputChannels = 5;
constexpr int kOutputWidth = 76;
constexpr int kOutputHeight = 512;
constexpr int kOutputHeatmapChannels = 1;
constexpr int kOutputOffsetChannels = 2;
constexpr std::size_t kInputFloatCount =
    static_cast<std::size_t>(kInputWidth) * kInputHeight * kInputChannels;
constexpr std::size_t kOutputHeatmapFloatCount =
    static_cast<std::size_t>(kOutputWidth) * kOutputHeight * kOutputHeatmapChannels;
constexpr std::size_t kOutputOffsetFloatCount =
    static_cast<std::size_t>(kOutputWidth) * kOutputHeight * kOutputOffsetChannels;
constexpr char kLogTag[] = "NcnnTinyHeatmap";
// NCNN's get_masked_option() uses bit 4 to disable Vulkan/tensor storage for a layer.
constexpr int kDisableVulkanFeatureMask = 1 << 4;

struct NcnnContext {
    ncnn::Net net;
    bool vulkan = false;
    // NCNN does not expose a per-layer runtime placement trace through this API.  A
    // support_vulkan flag is only a capability preflight and must not become a GPU claim.
    bool runtime_placement_verified = false;
    std::string placement_summary = "unverified";
    std::mutex run_mutex;
};

std::mutex g_gpu_mutex;
int g_gpu_users = 0;

void release_gpu_user() {
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_users <= 0) {
        return;
    }
    --g_gpu_users;
    if (g_gpu_users == 0) {
        ncnn::destroy_gpu_instance();
    }
}

bool acquire_gpu_user() {
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (g_gpu_users == 0 && ncnn::create_gpu_instance() != 0) {
        return false;
    }
    if (ncnn::get_gpu_count() <= 0 || ncnn::get_gpu_device(0) == nullptr) {
        if (g_gpu_users == 0) {
            ncnn::destroy_gpu_instance();
        }
        return false;
    }
    ++g_gpu_users;
    return true;
}

void destroy_context_after_failed_create(NcnnContext* context) {
    const bool vulkan = context->vulkan;
    // Release graph-owned Vulkan resources while the global device is still alive.
    context->net.clear();
    delete context;
    if (vulkan) {
        release_gpu_user();
    }
}

bool preflight_vulkan_layers(NcnnContext* context) {
    const auto& layers = context->net.layers();
    std::size_t active_layers = 0;
    std::size_t unsupported_layers = 0;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const ncnn::Layer* layer = layers[index];
        if (layer == nullptr) {
            ++unsupported_layers;
            __android_log_print(
                ANDROID_LOG_ERROR,
                kLogTag,
                "Vulkan capability preflight rejected layer=%zu reason=null_layer",
                index
            );
            continue;
        }
        if (layer->typeindex == ncnn::LayerType::Input) {
            continue;
        }
        ++active_layers;
        const bool disabled_by_mask = (layer->featmask & kDisableVulkanFeatureMask) != 0;
        if (layer->support_vulkan && !disabled_by_mask) {
            continue;
        }
        ++unsupported_layers;
        __android_log_print(
            ANDROID_LOG_ERROR,
            kLogTag,
            "Vulkan capability preflight rejected layer=%zu name=%s type=%s support_vulkan=%d "
            "featmask=0x%x disabled_by_mask=%d",
            index,
            layer->name.empty() ? "<unnamed>" : layer->name.c_str(),
            layer->type.empty() ? "<unknown>" : layer->type.c_str(),
            layer->support_vulkan ? 1 : 0,
            layer->featmask,
            disabled_by_mask ? 1 : 0
        );
    }

    if (active_layers == 0 || unsupported_layers != 0) {
        context->placement_summary =
            "vulkan_capability_preflight_rejected;active_layers=" +
            std::to_string(active_layers) + ";unsupported_layers=" +
            std::to_string(unsupported_layers) + ";runtime_placement=unverified";
        return false;
    }

    // This is intentionally not placement evidence.  NCNN's public Net/Layer API tells us
    // which implementation can use Vulkan, but not which implementation actually ran on every
    // layer after pipeline creation.  A future runtime trace or an equivalent device profile
    // must set runtime_placement_verified before production may report GPU success.
    context->placement_summary =
        "vulkan_capability_preflight_passed;active_layers=" +
        std::to_string(active_layers) + ";vulkan_capable_layers=" +
        std::to_string(active_layers) + ";runtime_placement=unverified";
    return true;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_cupet_fallingball_vision_NcnnTinyHeatmapNet_nativeCreate(
    JNIEnv* env,
    jobject /* this_object */,
    jobject asset_manager,
    jstring param_asset,
    jstring model_asset,
    jboolean use_vulkan
) {
    auto* manager = AAssetManager_fromJava(env, asset_manager);
    if (manager == nullptr || param_asset == nullptr || model_asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "invalid asset manager or names");
        return 0;
    }

    auto* context = new NcnnContext();
    context->vulkan = use_vulkan == JNI_TRUE;
    if (context->vulkan && !acquire_gpu_user()) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "Vulkan GPU instance unavailable");
        delete context;
        return 0;
    }

    context->net.opt.use_vulkan_compute = context->vulkan;
    context->net.opt.use_fp16_packed = context->vulkan;
    context->net.opt.use_fp16_storage = context->vulkan;
    context->net.opt.use_fp16_arithmetic = context->vulkan;
    context->net.opt.use_packing_layout = true;
    context->net.opt.lightmode = true;

    const char* param_name = env->GetStringUTFChars(param_asset, nullptr);
    const char* model_name = env->GetStringUTFChars(model_asset, nullptr);
    if (param_name == nullptr || model_name == nullptr) {
        if (param_name != nullptr) {
            env->ReleaseStringUTFChars(param_asset, param_name);
        }
        if (model_name != nullptr) {
            env->ReleaseStringUTFChars(model_asset, model_name);
        }
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "unable to read NCNN asset names");
        destroy_context_after_failed_create(context);
        return 0;
    }
    const int param_result = context->net.load_param(manager, param_name);
    const int model_result = param_result == 0 ? context->net.load_model(manager, model_name) : -1;
    env->ReleaseStringUTFChars(param_asset, param_name);
    env->ReleaseStringUTFChars(model_asset, model_name);

    if (param_result != 0 || model_result != 0) {
        __android_log_print(
            ANDROID_LOG_ERROR,
            kLogTag,
            "failed to load NCNN model param=%d model=%d",
            param_result,
            model_result
        );
        destroy_context_after_failed_create(context);
        return 0;
    }

    if (context->vulkan) {
        if (!preflight_vulkan_layers(context)) {
            __android_log_print(
                ANDROID_LOG_ERROR,
                kLogTag,
                "Vulkan session rejected before runtime: placement=%s",
                context->placement_summary.c_str()
            );
            destroy_context_after_failed_create(context);
            return 0;
        }
        const auto* device = ncnn::get_gpu_device(0);
        __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "Vulkan capability preflight passed runtime_placement_verified=%d summary=%s "
            "device=%s fp16_storage=%d fp16_arithmetic=%d",
            context->runtime_placement_verified ? 1 : 0,
            context->placement_summary.c_str(),
            device == nullptr ? "unknown" : device->info.device_name(),
            device != nullptr && device->info.support_fp16_storage() ? 1 : 0,
            device != nullptr && device->info.support_fp16_arithmetic() ? 1 : 0
        );
    } else {
        context->runtime_placement_verified = true;
        context->placement_summary = "cpu_full_graph";
        __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "NCNN running on CPU runtime_placement_verified=1 summary=%s",
            context->placement_summary.c_str()
        );
    }
    return reinterpret_cast<jlong>(context);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_cupet_fallingball_vision_NcnnTinyHeatmapNet_nativePlacementVerified(
    JNIEnv* /* env */,
    jobject /* this_object */,
    jlong handle
) {
    const auto* context = reinterpret_cast<const NcnnContext*>(handle);
    return context != nullptr && context->runtime_placement_verified ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_cupet_fallingball_vision_NcnnTinyHeatmapNet_nativePlacementSummary(
    JNIEnv* env,
    jobject /* this_object */,
    jlong handle
) {
    const auto* context = reinterpret_cast<const NcnnContext*>(handle);
    if (context == nullptr) {
        return env->NewStringUTF("unverified");
    }
    return env->NewStringUTF(context->placement_summary.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_cupet_fallingball_vision_NcnnTinyHeatmapNet_nativeRun(
    JNIEnv* env,
    jobject /* this_object */,
    jlong handle,
    jobject input_buffer,
    jobject heatmap_buffer,
    jobject offset_buffer
) {
    auto* context = reinterpret_cast<NcnnContext*>(handle);
    if (context == nullptr || input_buffer == nullptr || heatmap_buffer == nullptr ||
        offset_buffer == nullptr) {
        return JNI_FALSE;
    }
    const jlong input_capacity = env->GetDirectBufferCapacity(input_buffer);
    const jlong heatmap_capacity = env->GetDirectBufferCapacity(heatmap_buffer);
    const jlong offset_capacity = env->GetDirectBufferCapacity(offset_buffer);
    if (input_capacity < 0 || heatmap_capacity < 0 || offset_capacity < 0 ||
        static_cast<std::size_t>(input_capacity) < kInputFloatCount * sizeof(float) ||
        static_cast<std::size_t>(heatmap_capacity) < kOutputHeatmapFloatCount * sizeof(float) ||
        static_cast<std::size_t>(offset_capacity) < kOutputOffsetFloatCount * sizeof(float)) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "direct buffer capacity is too small");
        return JNI_FALSE;
    }
    auto* input = static_cast<const float*>(env->GetDirectBufferAddress(input_buffer));
    auto* heatmap = static_cast<float*>(env->GetDirectBufferAddress(heatmap_buffer));
    auto* offset = static_cast<float*>(env->GetDirectBufferAddress(offset_buffer));
    if (input == nullptr || heatmap == nullptr || offset == nullptr) {
        return JNI_FALSE;
    }

    std::lock_guard<std::mutex> lock(context->run_mutex);
    ncnn::Extractor extractor = context->net.create_extractor();
    extractor.set_light_mode(true);
    ncnn::Mat input_mat(
        kInputWidth,
        kInputHeight,
        kInputChannels,
        const_cast<float*>(input),
        sizeof(float)
    );
    if (extractor.input("in0", input_mat) != 0) {
        return JNI_FALSE;
    }

    ncnn::Mat heatmap_mat;
    ncnn::Mat offset_mat;
    // Explicit type=0 asks NCNN to convert fp16 storage and packed Vulkan tensors back to
    // ordinary float32/pack1 Mats before the JNI copy. Without this, a future optimizer change
    // could make row() expose packed or half-precision bytes while the Java side still reads
    // FloatBuffers, silently corrupting detections.
    if (extractor.extract("out0", heatmap_mat, 0) != 0 ||
        extractor.extract("out1", offset_mat, 0) != 0) {
        return JNI_FALSE;
    }
    if (heatmap_mat.w != kOutputWidth || heatmap_mat.h != kOutputHeight ||
        heatmap_mat.c != kOutputHeatmapChannels ||
        heatmap_mat.elempack != 1 || heatmap_mat.elemsize != sizeof(float) ||
        offset_mat.w != kOutputWidth || offset_mat.h != kOutputHeight ||
        offset_mat.c != kOutputOffsetChannels || offset_mat.elempack != 1 ||
        offset_mat.elemsize != sizeof(float)) {
        return JNI_FALSE;
    }

    const std::size_t heatmap_size =
        static_cast<std::size_t>(kOutputWidth) * kOutputHeight;
    for (int channel = 0; channel < kOutputHeatmapChannels; ++channel) {
        const ncnn::Mat channel_mat = heatmap_mat.channel(channel);
        std::memcpy(heatmap + channel * heatmap_size, channel_mat.row(0), heatmap_size * sizeof(float));
    }
    for (int channel = 0; channel < kOutputOffsetChannels; ++channel) {
        const ncnn::Mat channel_mat = offset_mat.channel(channel);
        std::memcpy(offset + channel * heatmap_size, channel_mat.row(0), heatmap_size * sizeof(float));
    }
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_cupet_fallingball_vision_NcnnTinyHeatmapNet_nativeDestroy(
    JNIEnv* /* env */,
    jobject /* this_object */,
    jlong handle
) {
    auto* context = reinterpret_cast<NcnnContext*>(handle);
    if (context == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(context->run_mutex);
        context->net.clear();
    }
    const bool vulkan = context->vulkan;
    delete context;
    if (vulkan) {
        release_gpu_user();
    }
}
