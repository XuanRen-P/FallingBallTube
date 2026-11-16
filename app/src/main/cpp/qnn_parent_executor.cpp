// Qualcomm HTP executor for the fixed-shape TinyHeatmapNet parent graph.
//
// Loads the QNN runtime (libQnnHtp.so) and metadata reader (libQnnSystem.so)
// by name from the APK's native library directory, deserializes one compiled
// context binary (AI Hub `qnn_context_binary`, HTP FP16), and executes the
// single retrieved graph.  One live instance is supported at a time; the
// Kotlin side owns the lifecycle and falls back to the ORT CPU backend when
// any step fails.
//
// I/O contract is validated at load time against the frozen production
// contract ([1,5,1024,152] float32 in, [1,1,512,76] + [1,2,512,76] float32
// out).  A mismatch aborts construction instead of producing garbage.

#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/System/QnnSystemContext.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpDeviceConfigShared.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"

#include <sys/system_properties.h>

#define LOG_TAG "TinyHeatmapNet"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

constexpr uint32_t kInputBatch = 1;
constexpr uint32_t kInputChannels = 9;
constexpr uint32_t kInputHeight = 1024;
constexpr uint32_t kInputWidth = 152;
constexpr uint32_t kHeatmapHeight = 512;
constexpr uint32_t kHeatmapWidth = 76;

constexpr uint32_t kInputRank = 4;
constexpr uint32_t kOutputRank = 4;

using QnnInterfaceGetProvidersFn_t =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t*** providerList, uint32_t* numProviders);
using QnnSystemGetProvidersFn_t =
    Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

struct GraphTensors {
    // Deep copies: the metadata returned by QnnSystemContext is owned by QNN.
    std::vector<Qnn_Tensor_t> tensors;
    std::vector<std::vector<uint32_t>> dims;
};

struct QnnState {
    void* htpLib = nullptr;
    void* systemLib = nullptr;
    const QnnInterface_t* provider = nullptr;
    Qnn_LogHandle_t log = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    GraphTensors inputs;
    GraphTensors outputs;
    std::string scannedGraphName;
    uint32_t powerConfigId = 0;
    bool ready = false;
};

QnnState g_state;

// QnnSystemContext interface table; only three members are needed and they
// sit at the start of the versioned struct.  Resolve through the versioned
// member the same way the HTP interface is resolved.
struct SystemContextApi {
    QnnSystemContext_CreateFn_t create = nullptr;
    QnnSystemContext_GetBinaryInfoFn_t getBinaryInfo = nullptr;
    QnnSystemContext_FreeFn_t freeFn = nullptr;
};

SystemContextApi g_systemApi;

void qnnLogCallback(const char* format,
                    QnnLog_Level_t /*level*/,
                    uint64_t /*timestamp*/,
                    va_list args) {
    // Surface QNN's own runtime messages at every level: HTP device/driver
    // failures are otherwise invisible because they bypass system logging.
    char message[1024];
    vsnprintf(message, sizeof(message), format != nullptr ? format : "(null)", args);
    __android_log_print(ANDROID_LOG_WARN, "TinyHeatmapNet", "QNN: %s", message);
}

// Logs the backend's human-readable reason for a failed call.  QNN error
// handles are the non-zero status values returned by the failed call itself.
void logQnnError(const char* what, Qnn_ErrorHandle_t error) {
    if (g_state.provider == nullptr || error == 0) return;
    const char* message = nullptr;
    if (QNN_SUCCESS == g_state.provider->QNN_INTERFACE_VER_NAME.errorGetMessage(error,
                                                                               &message) &&
        message != nullptr) {
        LOGE("%s: %s", what, message);
    } else {
        LOGE("%s: (no error message available, status=%lu)", what,
             static_cast<unsigned long>(error));
    }
}

// QNN SDK >= 2.28 requires apps to enable unsigned PD explicitly before HTP
// device init (Qualcomm AI Engine Direct FAQ); without it the transport fails
// to load the unsigned skel.  Measured on SM8650: loadRemoteSymbols err 4000.
// The camera HAL on the same device runs this exact call (domain 7) before
// successfully opening fastrpc_shell_unsigned_3.
//
// The app linker namespace refuses both the vendor libcdsprpc.so by absolute
// path and a bundled copy (it needs libhidlbase, which apps cannot link), so
// grab the copy the HTP stub already loaded as its own dependency: this must
// run AFTER backendCreate and BEFORE deviceCreate.
void enableUnsignedPd() {
    void* cdspRpc = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_NOLOAD);
    if (cdspRpc == nullptr) {
        LOGW("unsigned PD: libcdsprpc.so not loaded in process; skipping");
        return;
    }
    using RemoteSessionControlFn = int (*)(unsigned int control, void* payload,
                                           unsigned int domain);
    auto remoteSessionControl = reinterpret_cast<RemoteSessionControlFn>(
        dlsym(cdspRpc, "remote_session_control"));
    if (remoteSessionControl == nullptr) {
        LOGW("unsigned PD: libcdsprpc.so has no remote_session_control symbol");
        return;
    }
    constexpr unsigned int kDspUnsignedPdDomain = 4;  // Hexagon SDK remote.h
    // Domain candidates: 7 matched this device's camera HAL trace; 3 is the
    // classic CDSP domain id on earlier SoCs.
    for (unsigned int domain : {7u, 3u}) {
        unsigned int enable = 1;
        const int result =
            remoteSessionControl(kDspUnsignedPdDomain, &enable, domain);
        LOGI("remote_session_control(unsigned PD enable, domain %u) -> %d", domain,
             result);
        if (result == 0) break;
    }
}

// Votes the HTP out of its default low-power DCVS state.  Without this the
// device runs the graph at saver clocks (measured: 100 ms for a 19 ms graph).
void applyHtpBurstMode() {
    const auto& iface = g_state.provider->QNN_INTERFACE_VER_NAME;
    QnnDevice_Infrastructure_t infrastructure = nullptr;
    if (QNN_SUCCESS != iface.deviceGetInfrastructure(&infrastructure) ||
        infrastructure == nullptr) {
        LOGW("HTP perf: deviceGetInfrastructure unavailable");
        return;
    }
    auto* htpInfrastructure = static_cast<QnnHtpDevice_Infrastructure_t*>(infrastructure);
    if (htpInfrastructure->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
        LOGW("HTP perf: unexpected infrastructure type %d",
             static_cast<int>(htpInfrastructure->infraType));
        return;
    }
    uint32_t powerConfigId = 0;
    if (QNN_SUCCESS != htpInfrastructure->perfInfra.createPowerConfigId(0, 0,
                                                                       &powerConfigId)) {
        LOGW("HTP perf: createPowerConfigId failed");
        return;
    }
    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig{};
    powerConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& dcvs = powerConfig.dcvsV3Config;
    // Match the payload client to setPowerConfig, as Qualcomm's boostPerformance does:
    // https://github.com/qualcomm/qai-appbuilder/blob/main/src/QnnInferenceEngine.cpp
    dcvs.contextId = powerConfigId;
    // Mirrors the qnn-net-run burst profile: DCVS disabled, sleep disabled,
    // every voltage corner voted to the platform maximum.
    dcvs.setDcvsEnable = 1;
    dcvs.dcvsEnable = 0;
    dcvs.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvs.setSleepLatency = 1;
    dcvs.sleepLatency = 40;
    dcvs.setSleepDisable = 1;
    dcvs.sleepDisable = 1;
    dcvs.setBusParams = 1;
    dcvs.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.setCoreParams = 1;
    dcvs.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&powerConfig,
                                                                    nullptr};
    const Qnn_ErrorHandle_t powerStatus =
        htpInfrastructure->perfInfra.setPowerConfig(powerConfigId, powerConfigs);
    if (QNN_SUCCESS != powerStatus) {
        // SDK error 13001 is INVALID_INPUT, not evidence of a permission restriction.
        // Keep RPC tuning available even when the device rejects the DCVS request.
        LOGW("HTP perf: DCVS vote rejected (status=%lu, %s) - continuing with RPC tuning",
             static_cast<unsigned long>(powerStatus),
             powerStatus == QNN_HTP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT
                 ? "invalid input" : "power configuration failure");
    }
    // RPC polling keeps the DSP responsive between execute calls; without it
    // every inference pays a fixed wakeup round-trip.
    QnnHtpPerfInfrastructure_PowerConfig_t rpcControl{};
    rpcControl.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControl.rpcControlLatencyConfig = 100;  // microseconds
    QnnHtpPerfInfrastructure_PowerConfig_t rpcPolling{};
    rpcPolling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPolling.rpcPollingTimeConfig = 9999;  // microseconds
    const QnnHtpPerfInfrastructure_PowerConfig_t* rpcConfigs[] = {&rpcControl,
                                                                  &rpcPolling,
                                                                  nullptr};
    if (QNN_SUCCESS != htpInfrastructure->perfInfra.setPowerConfig(powerConfigId,
                                                                  rpcConfigs)) {
        LOGW("HTP perf: RPC latency/polling config failed");
        return;
    }
    g_state.powerConfigId = powerConfigId;
    LOGI("HTP performance config: DCVS=%s; RPC latency 100us, polling 9999us",
         powerStatus == QNN_SUCCESS ? "accepted" : "rejected");
}

// HTP architecture + SoC identity for the platforms we ship context binaries for.
// A context binary is bound to one Hexagon arch: a v75 binary fails to load on a v73
// device (measured on a Xiaomi 13 / SM8550: loadRemoteSymbols err 4000 -> transport
// 14001 -> INVALID_CONFIG), so the offline platform information has to describe the
// device it is actually running on rather than a hard-coded 8 Gen 3.
struct SocPlatform {
    uint32_t socModel;
    QnnHtpDevice_Arch_t arch;
    uint32_t vtcmMb;
};

// Maps ro.soc.model to (soc model id, HTP arch).  OEM SKU suffixes such as the Galaxy
// "-AC" part are ignored: the HTP arch is the ABI that decides binary compatibility.
bool socPlatformFromSystemProperties(SocPlatform* out) {
    char value[PROP_VALUE_MAX] = {0};
    int length = __system_property_get("ro.soc.model", value);
    const std::string socModel(length > 0 ? value : "");
    struct Entry {
        const char* prefix;
        uint32_t socModel;
        QnnHtpDevice_Arch_t arch;
    };
    static const Entry kEntries[] = {
        {"SM8450", QNN_SOC_MODEL_SM8450, QNN_HTP_DEVICE_ARCH_V69},  // 8 Gen 1
        {"SM8475", QNN_SOC_MODEL_SM8475, QNN_HTP_DEVICE_ARCH_V69},  // 8+ Gen 1
        {"SM8550", QNN_SOC_MODEL_SM8550, QNN_HTP_DEVICE_ARCH_V73},  // 8 Gen 2
        {"SM8635", QNN_SOC_MODEL_SM8635, QNN_HTP_DEVICE_ARCH_V73},  // 8s Gen 3
        {"SM8650", QNN_SOC_MODEL_SM8650, QNN_HTP_DEVICE_ARCH_V75},  // 8 Gen 3
        {"SM8750", QNN_SOC_MODEL_SM8750, QNN_HTP_DEVICE_ARCH_V79},  // 8 Elite (incl. -AC)
    };
    for (const Entry& entry : kEntries) {
        if (socModel.rfind(entry.prefix, 0) != 0) continue;
        out->socModel = entry.socModel;
        out->arch = entry.arch;
        out->vtcmMb = 8;
        return true;
    }
    return false;
}

// Second-chance device creation: builds an explicit offline platform info so
// QNN never needs the sysfs SoC detection that SELinux denies untrusted apps
// on HyperOS/MIUI builds (measured: avc denial on /sys/devices/soc0/sku
// immediately before QnnDevice_create failed).  Only reached when the normal
// auto-detecting deviceCreate failed.
bool createDeviceWithOfflineInfo(Qnn_LogHandle_t log, Qnn_DeviceHandle_t* device) {
    SocPlatform platform{};
    if (!socPlatformFromSystemProperties(&platform)) {
        LOGE("QnnDevice_create failed and no offline SoC mapping is available "
             "for this device");
        return false;
    }
    QnnHtpDevice_DeviceInfoExtension_t htpExtension{};
    htpExtension.devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
    htpExtension.onChipDevice.vtcmSize = platform.vtcmMb;
    htpExtension.onChipDevice.socModel = platform.socModel;
    htpExtension.onChipDevice.signedPdSupport = false;
    htpExtension.onChipDevice.dlbcSupport = true;
    htpExtension.onChipDevice.arch = platform.arch;

    QnnDevice_HardwareDeviceInfo_t hardwareInfo = QNN_DEVICE_HARDWARE_DEVICE_INFO_INIT;
    hardwareInfo.v1.deviceId = 0;  // measured: platform supports device ids 0..1
    hardwareInfo.v1.deviceType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
    // measured: platform validates exactly one core for this device
    QnnDevice_CoreInfo_t coreInfo = QNN_DEVICE_CORE_INFO_INIT;
    coreInfo.v1.coreId = 0;
    coreInfo.v1.coreType = QNN_HTP_CORE_TYPE_NSP;  // HTP is an NSP-family core
    hardwareInfo.v1.numCores = 1;
    hardwareInfo.v1.cores = &coreInfo;
    hardwareInfo.v1.deviceInfoExtension =
        reinterpret_cast<QnnDevice_DeviceInfoExtension_t>(&htpExtension);

    QnnDevice_PlatformInfo_t platformInfo = QNN_DEVICE_PLATFORM_INFO_INIT;
    platformInfo.v1.numHwDevices = 1;
    platformInfo.v1.hwDevices = &hardwareInfo;

    QnnDevice_Config_t deviceConfig = QNN_DEVICE_CONFIG_INIT;
    deviceConfig.option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
    deviceConfig.hardwareInfo = &platformInfo;
    const QnnDevice_Config_t* deviceConfigs[] = {&deviceConfig, nullptr};

    LOGI("Retrying QnnDevice_create with offline platform info (socModel=%u arch=%d)",
         platform.socModel, static_cast<int>(platform.arch));
    const auto& iface = g_state.provider->QNN_INTERFACE_VER_NAME;
    const Qnn_ErrorHandle_t status = iface.deviceCreate(log, deviceConfigs, device);
    if (QNN_SUCCESS != status) {
        logQnnError("QnnDevice_create with offline info failed", status);
        return false;
    }
    return true;
}

// Normalizes graph metadata tensors (V1 or V2) into execution tensors of
// version V2, which the 2.39 interface accepts for both.
bool copyTensorTemplate(const Qnn_Tensor_t& source,
                        std::vector<Qnn_Tensor_t>& outTensors,
                        std::vector<std::vector<uint32_t>>& outDims) {
    uint32_t id = 0;
    uint32_t rank = 0;
    const uint32_t* dimensions = nullptr;
    Qnn_TensorType_t type = QNN_TENSOR_TYPE_UNDEFINED;
    Qnn_TensorDataFormat_t dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    Qnn_DataType_t dataType = QNN_DATATYPE_UNDEFINED;
    if (source.version == QNN_TENSOR_VERSION_1) {
        id = source.v1.id;
        rank = source.v1.rank;
        dimensions = source.v1.dimensions;
        type = source.v1.type;
        dataFormat = source.v1.dataFormat;
        dataType = source.v1.dataType;
    } else if (source.version == QNN_TENSOR_VERSION_2) {
        id = source.v2.id;
        rank = source.v2.rank;
        dimensions = source.v2.dimensions;
        type = source.v2.type;
        dataFormat = source.v2.dataFormat;
        dataType = source.v2.dataType;
    } else {
        LOGE("Unsupported tensor metadata version %d", static_cast<int>(source.version));
        return false;
    }
    outTensors.emplace_back();
    Qnn_Tensor_t& copy = outTensors.back();
    copy.version = QNN_TENSOR_VERSION_2;
    copy.v2 = QNN_TENSOR_V2_INIT;
    copy.v2.id = id;
    copy.v2.type = type;
    copy.v2.dataFormat = dataFormat;
    copy.v2.dataType = dataType;
    copy.v2.rank = rank;
    outDims.emplace_back();
    std::vector<uint32_t>& owned = outDims.back();
    if (rank > 0) {
        owned.assign(dimensions, dimensions + rank);
    }
    copy.v2.dimensions = rank > 0 ? owned.data() : nullptr;
    copy.v2.memType = QNN_TENSORMEMTYPE_RAW;
    copy.v2.clientBuf = Qnn_ClientBuffer_t{nullptr, 0u};
    return true;
}

bool dimsMatch(const std::vector<uint32_t>& dims,
               std::initializer_list<uint32_t> expected) {
    if (dims.size() != expected.size()) return false;
    size_t index = 0;
    for (uint32_t value : expected) {
        if (dims[index] != value) return false;
        ++index;
    }
    return true;
}

bool isFloat32(const Qnn_Tensor_t& tensor) {
    return tensor.v2.dataType == QNN_DATATYPE_FLOAT_32;
}

// Finds a plausible compiled graph name ("graph_<suffix>") inside the context
// binary.  AI Hub-generated binaries carry the name in plain ASCII even when
// the SystemContext metadata leaves it empty.
std::string scanGraphName(const uint8_t* begin, const uint8_t* end) {
    const char* key = "graph_";
    const size_t keyLength = 6;
    auto nameChar = [](uint8_t c) {
        return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    };
    for (const uint8_t* cursor = begin; cursor + keyLength < end; ++cursor) {
        if (std::memcmp(cursor, key, keyLength) != 0) continue;
        const uint8_t* nameStart = cursor;
        const uint8_t* scan = cursor + keyLength;
        while (scan < end && nameChar(*scan)) ++scan;
        if (scan > cursor + keyLength) {
            return std::string(reinterpret_cast<const char*>(nameStart),
                               static_cast<size_t>(scan - nameStart));
        }
        cursor = scan;
    }
    return {};
}

// Version-agnostic view over QnnSystemContext_GraphInfo_t.
struct GraphMetaView {
    const char* graphName;
    uint32_t numGraphInputs;
    const Qnn_Tensor_t* graphInputs;
    uint32_t numGraphOutputs;
    const Qnn_Tensor_t* graphOutputs;
};

bool graphMetaView(const QnnSystemContext_GraphInfo_t& wrapper, GraphMetaView& view) {
    switch (wrapper.version) {
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: {
            const auto& v = wrapper.graphInfoV1;
            view = {v.graphName, v.numGraphInputs, v.graphInputs, v.numGraphOutputs,
                    v.graphOutputs};
            return true;
        }
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: {
            const auto& v = wrapper.graphInfoV2;
            view = {v.graphName, v.numGraphInputs, v.graphInputs, v.numGraphOutputs,
                    v.graphOutputs};
            return true;
        }
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: {
            const auto& v = wrapper.graphInfoV3;
            view = {v.graphName, v.numGraphInputs, v.graphInputs, v.numGraphOutputs,
                    v.graphOutputs};
            return true;
        }
        default:
            LOGE("Unsupported graph metadata version %d", static_cast<int>(wrapper.version));
            return false;
    }
}

void destroyLocked() {
    const auto& iface = g_state.provider->QNN_INTERFACE_VER_NAME;
    if (g_state.context != nullptr) {
        iface.contextFree(g_state.context, nullptr);
        g_state.context = nullptr;
        g_state.graph = nullptr;
    }
    if (g_state.device != nullptr && g_state.powerConfigId != 0) {
        // Best-effort release of the power vote taken by applyHtpBurstMode.
        QnnDevice_Infrastructure_t infrastructure = nullptr;
        if (QNN_SUCCESS == iface.deviceGetInfrastructure(&infrastructure) &&
            infrastructure != nullptr) {
            auto* htpInfrastructure =
                static_cast<QnnHtpDevice_Infrastructure_t*>(infrastructure);
            if (htpInfrastructure->infraType ==
                QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
                htpInfrastructure->perfInfra.destroyPowerConfigId(
                    g_state.powerConfigId);
            }
        }
        g_state.powerConfigId = 0;
    }
    if (g_state.device != nullptr) {
        iface.deviceFree(g_state.device);
        g_state.device = nullptr;
    }
    if (g_state.backend != nullptr) {
        iface.backendFree(g_state.backend);
        g_state.backend = nullptr;
    }
    if (g_state.log != nullptr) {
        iface.logFree(g_state.log);
        g_state.log = nullptr;
    }
    g_state.ready = false;
}

// Single-library load: take the only provider the runtime advertises.  The
// runtime lib version matches the headers we compiled against.
const QnnInterface_t* selectProvider(const QnnInterface_t* const* providers,
                                     uint32_t numProviders) {
    for (uint32_t index = 0; index < numProviders; ++index) {
        const Qnn_Version_t& version = providers[index]->apiVersion.coreApiVersion;
        LOGI("QNN provider %u: core apiVersion %u.%u.%u", index, version.major, version.minor,
             version.patch);
    }
    return numProviders > 0 ? providers[0] : nullptr;
}

bool initSystemApi() {
    g_state.systemLib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (g_state.systemLib == nullptr) {
        LOGE("dlopen(libQnnSystem.so) failed: %s", dlerror());
        return false;
    }
    auto getProviders = reinterpret_cast<QnnSystemGetProvidersFn_t>(
        dlsym(g_state.systemLib, "QnnSystemInterface_getProviders"));
    if (getProviders == nullptr) {
        LOGE("dlsym(QnnSystemInterface_getProviders) failed: %s", dlerror());
        return false;
    }
    const QnnSystemInterface_t** systemProviders = nullptr;
    uint32_t numSystemProviders = 0;
    if (QNN_SUCCESS != getProviders(&systemProviders, &numSystemProviders) ||
        systemProviders == nullptr || numSystemProviders == 0) {
        LOGE("QnnSystemInterface_getProviders returned no providers");
        return false;
    }
    const auto& table = systemProviders[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    g_systemApi.create = table.systemContextCreate;
    g_systemApi.getBinaryInfo = table.systemContextGetBinaryInfo;
    g_systemApi.freeFn = table.systemContextFree;
    if (g_systemApi.create == nullptr || g_systemApi.getBinaryInfo == nullptr ||
        g_systemApi.freeFn == nullptr) {
        LOGE("QNN system interface table is incomplete");
        return false;
    }
    return true;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_cupet_fallingball_vision_QnnParentRuntime_nativeCreateFromBuffer(
    JNIEnv* env, jobject /*thiz*/, jstring htpLibraryPath, jbyteArray contextBinary) {
    if (g_state.ready) return JNI_TRUE;
    // A previously failed attempt may have left partial metadata behind.
    g_state.inputs.tensors.clear();
    g_state.inputs.dims.clear();
    g_state.outputs.tensors.clear();
    g_state.outputs.dims.clear();

    const char* htpPath = env->GetStringUTFChars(htpLibraryPath, nullptr);
    if (htpPath == nullptr) return JNI_FALSE;
    g_state.htpLib = dlopen(htpPath, RTLD_NOW | RTLD_LOCAL);
    // fastRPC resolves the Hexagon-side skel by searching ADSP_LIBRARY_PATH; an
    // app-installed unsigned skel is only visible once its lib dir is listed.
    if (const char* slash = std::strrchr(htpPath, '/'); slash != nullptr) {
        std::string libDir(htpPath, static_cast<size_t>(slash - htpPath));
        std::string searchPath = libDir + ";/system/lib/rfsa/adsp;/vendor/dsp/cdsp;"
                                         "/vendor/lib/rfsa/adsp;/vendor/dsp/registry";
        setenv("ADSP_LIBRARY_PATH", searchPath.c_str(), 1);
    }
    env->ReleaseStringUTFChars(htpLibraryPath, htpPath);
    if (g_state.htpLib == nullptr) {
        LOGE("dlopen(htp) failed: %s", dlerror());
        return JNI_FALSE;
    }

    auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(
        dlsym(g_state.htpLib, "QnnInterface_getProviders"));
    if (getProviders == nullptr) {
        LOGE("dlsym(QnnInterface_getProviders) failed: %s", dlerror());
        return JNI_FALSE;
    }
    const QnnInterface_t** providerPtrs = nullptr;
    uint32_t numProviders = 0;
    Qnn_ErrorHandle_t providersStatus = getProviders(&providerPtrs, &numProviders);
    if (QNN_SUCCESS != providersStatus || providerPtrs == nullptr || numProviders == 0) {
        LOGE("QnnInterface_getProviders returned no providers (status=%d)",
             static_cast<int>(providersStatus));
        return JNI_FALSE;
    }
    g_state.provider = selectProvider(providerPtrs, numProviders);
    if (g_state.provider == nullptr) {
        LOGE("QNN runtime advertised no providers");
        return JNI_FALSE;
    }
    const auto& iface = g_state.provider->QNN_INTERFACE_VER_NAME;

    if (!initSystemApi()) return JNI_FALSE;
    LOGI("QNN system interface ready");

    const jsize binarySize = env->GetArrayLength(contextBinary);
    if (binarySize <= 0) {
        LOGE("Empty context binary");
        return JNI_FALSE;
    }
    std::vector<uint8_t> binary(static_cast<size_t>(binarySize));
    env->GetByteArrayRegion(contextBinary, 0, binarySize,
                            reinterpret_cast<jbyte*>(binary.data()));
    const uint8_t* binaryData = binary.data();

    // Read graph metadata first: the graph name and I/O tensor templates.
    QnnSystemContext_Handle_t systemContext = nullptr;
    if (QNN_SUCCESS != g_systemApi.create(&systemContext) || systemContext == nullptr) {
        LOGE("QnnSystemContext_create failed");
        return JNI_FALSE;
    }
    const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
    Qnn_ContextBinarySize_t binaryInfoSize = 0;
    Qnn_ErrorHandle_t infoStatus = g_systemApi.getBinaryInfo(
        systemContext, const_cast<uint8_t*>(binaryData), static_cast<uint64_t>(binarySize),
        &binaryInfo, &binaryInfoSize);
    if (QNN_SUCCESS != infoStatus || binaryInfo == nullptr) {
        LOGE("QnnSystemContext_getBinaryInfo failed (status=%d)",
             static_cast<int>(infoStatus));
        g_systemApi.freeFn(systemContext);
        return JNI_FALSE;
    }
    // All BinaryInfo versions end their layout with numGraphs/graphs before the
    // optional tails, but dispatch explicitly rather than relying on prefixes.
    const QnnSystemContext_GraphInfo_t* graphInfos = nullptr;
    uint32_t numGraphs = 0;
    switch (binaryInfo->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            numGraphs = binaryInfo->contextBinaryInfoV1.numGraphs;
            graphInfos = binaryInfo->contextBinaryInfoV1.graphs;
            break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            numGraphs = binaryInfo->contextBinaryInfoV2.numGraphs;
            graphInfos = binaryInfo->contextBinaryInfoV2.graphs;
            break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            numGraphs = binaryInfo->contextBinaryInfoV3.numGraphs;
            graphInfos = binaryInfo->contextBinaryInfoV3.graphs;
            break;
        default:
            LOGE("Unsupported context binary metadata version %d",
                 static_cast<int>(binaryInfo->version));
            g_systemApi.freeFn(systemContext);
            return JNI_FALSE;
    }
    if (numGraphs != 1 || graphInfos == nullptr) {
        LOGE("Expected exactly 1 graph, found %u", numGraphs);
        g_systemApi.freeFn(systemContext);
        return JNI_FALSE;
    }
    GraphMetaView graphMeta{};
    if (!graphMetaView(graphInfos[0], graphMeta)) {
        g_systemApi.freeFn(systemContext);
        return JNI_FALSE;
    }
    const char* graphName = graphMeta.graphName;
    LOGI("Metadata graph name: %s",
         graphName == nullptr ? "(null)"
                              : (graphName[0] == '\0' ? "(empty)" : graphName));
    // QAIRT-generated context binaries may carry an empty metadata graph name;
    // the compiled graph inside still has one ("graph_<job>", discoverable in
    // the binary itself).  Scan before giving up.
    if (graphName == nullptr || graphName[0] == '\0') {
        g_state.scannedGraphName = scanGraphName(binaryData, binaryData + binarySize);
        if (!g_state.scannedGraphName.empty()) {
            graphName = g_state.scannedGraphName.c_str();
            LOGI("Metadata graph name empty; using scanned graph name '%s'", graphName);
        }
    }
    if (graphMeta.numGraphInputs != 1 || graphMeta.numGraphOutputs != 2 ||
        graphName == nullptr || graphName[0] == '\0') {
        LOGE("Unexpected graph shape: name=%s inputs=%u outputs=%u", graphName == nullptr ? "null" : graphName,
             graphMeta.numGraphInputs, graphMeta.numGraphOutputs);
        g_systemApi.freeFn(systemContext);
        return JNI_FALSE;
    }
    if (!copyTensorTemplate(graphMeta.graphInputs[0], g_state.inputs.tensors, g_state.inputs.dims) ||
        !copyTensorTemplate(graphMeta.graphOutputs[0], g_state.outputs.tensors, g_state.outputs.dims) ||
        !copyTensorTemplate(graphMeta.graphOutputs[1], g_state.outputs.tensors, g_state.outputs.dims)) {
        g_systemApi.freeFn(systemContext);
        return JNI_FALSE;
    }
    // graphName points into QNN-owned metadata; freeFn invalidates it, so take
    // an owned copy before releasing the system context.
    if (g_state.scannedGraphName.empty() && graphName != nullptr) {
        g_state.scannedGraphName = graphName;
    }
    graphName = g_state.scannedGraphName.c_str();
    g_systemApi.freeFn(systemContext);
    LOGI("Context binary metadata parsed: inputs=%zu outputs=%zu", g_state.inputs.dims.size(),
         g_state.outputs.dims.size());
    if (g_state.inputs.dims[0].size() != kInputRank ||
        !dimsMatch(g_state.inputs.dims[0],
                   {kInputBatch, kInputChannels, kInputHeight, kInputWidth}) ||
        !isFloat32(g_state.inputs.tensors[0])) {
        const std::vector<uint32_t>& d = g_state.inputs.dims[0];
        LOGE("Input contract mismatch: rank=%zu dims=%u,%u,%u,%u dtype=%d", d.size(),
             d.size() > 0 ? d[0] : 0u, d.size() > 1 ? d[1] : 0u, d.size() > 2 ? d[2] : 0u,
             d.size() > 3 ? d[3] : 0u, static_cast<int>(g_state.inputs.tensors[0].v2.dataType));
        return JNI_FALSE;
    }
    if (g_state.outputs.dims[0].size() != kOutputRank ||
        !dimsMatch(g_state.outputs.dims[0], {1, 1, kHeatmapHeight, kHeatmapWidth}) ||
        g_state.outputs.dims[1].size() != kOutputRank ||
        !dimsMatch(g_state.outputs.dims[1], {1, 2, kHeatmapHeight, kHeatmapWidth}) ||
        !isFloat32(g_state.outputs.tensors[0]) || !isFloat32(g_state.outputs.tensors[1])) {
        LOGE("Output contract mismatch");
        return JNI_FALSE;
    }

    if (QNN_SUCCESS != iface.logCreate(&qnnLogCallback, QNN_LOG_LEVEL_ERROR, &g_state.log)) {
        LOGE("QnnLog_create failed");
        return JNI_FALSE;
    }
    LOGI("QNN log created");
    if (QNN_SUCCESS != iface.backendCreate(g_state.log, nullptr, &g_state.backend)) {
        LOGE("QnnBackend_create failed");
        destroyLocked();
        return JNI_FALSE;
    }
    LOGI("QNN backend created");
    // The HTP stub (loaded above with its libcdsprpc dependency) must already be
    // in the process before unsigned PD can be enabled.
    enableUnsignedPd();
    const Qnn_ErrorHandle_t deviceStatus = iface.deviceCreate(g_state.log, nullptr, &g_state.device);
    if (QNN_SUCCESS != deviceStatus) {
        logQnnError("QnnDevice_create failed", deviceStatus);
        // Retry with an explicit offline platform info before giving up; the
        // log/backend handles stay alive for the retry.
        if (!createDeviceWithOfflineInfo(g_state.log, &g_state.device)) {
            destroyLocked();
            return JNI_FALSE;
        }
    }
    LOGI("QNN device created");
    applyHtpBurstMode();
    if (QNN_SUCCESS != iface.contextCreateFromBinary(
                            g_state.backend, g_state.device, nullptr, binaryData,
                            static_cast<Qnn_ContextBinarySize_t>(binarySize), &g_state.context,
                            nullptr)) {
        LOGE("QnnContext_createFromBinary failed");
        destroyLocked();
        return JNI_FALSE;
    }
    if (QNN_SUCCESS != iface.graphRetrieve(g_state.context, graphName, &g_state.graph) ||
        g_state.graph == nullptr) {
        LOGE("QnnGraph_retrieve(%s) failed", graphName);
        destroyLocked();
        return JNI_FALSE;
    }
    g_state.ready = true;
    LOGI("QNN HTP context ready: graph=%s input=%ux%ux%ux%u", graphName, kInputBatch,
         kInputChannels, kInputHeight, kInputWidth);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_cupet_fallingball_vision_QnnParentRuntime_nativeExecute(
    JNIEnv* env, jobject /*thiz*/, jobject inputBuffer, jobject heatmapBuffer,
    jobject offsetBuffer) {
    if (!g_state.ready) return JNI_FALSE;
    auto* input = static_cast<float*>(env->GetDirectBufferAddress(inputBuffer));
    auto* heatmap = static_cast<float*>(env->GetDirectBufferAddress(heatmapBuffer));
    auto* offset = static_cast<float*>(env->GetDirectBufferAddress(offsetBuffer));
    if (input == nullptr || heatmap == nullptr || offset == nullptr) return JNI_FALSE;

    const auto& iface = g_state.provider->QNN_INTERFACE_VER_NAME;
    g_state.inputs.tensors[0].v2.clientBuf.data = input;
    g_state.inputs.tensors[0].v2.clientBuf.dataSize =
        sizeof(float) * kInputChannels * kInputHeight * kInputWidth;
    g_state.outputs.tensors[0].v2.clientBuf.data = heatmap;
    g_state.outputs.tensors[0].v2.clientBuf.dataSize =
        sizeof(float) * kHeatmapHeight * kHeatmapWidth;
    g_state.outputs.tensors[1].v2.clientBuf.data = offset;
    g_state.outputs.tensors[1].v2.clientBuf.dataSize =
        sizeof(float) * 2 * kHeatmapHeight * kHeatmapWidth;

    return QNN_SUCCESS == iface.graphExecute(g_state.graph,
                                             g_state.inputs.tensors.data(), 1,
                                             g_state.outputs.tensors.data(), 2, nullptr,
                                             nullptr)
               ? JNI_TRUE
               : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_cupet_fallingball_vision_QnnParentRuntime_nativeDestroy(JNIEnv* /*env*/,
                                                                 jobject /*thiz*/) {
    if (g_state.provider != nullptr) {
        destroyLocked();
    }
    g_state.provider = nullptr;
    if (g_state.systemLib != nullptr) {
        dlclose(g_state.systemLib);
        g_state.systemLib = nullptr;
    }
    if (g_state.htpLib != nullptr) {
        dlclose(g_state.htpLib);
        g_state.htpLib = nullptr;
    }
}
