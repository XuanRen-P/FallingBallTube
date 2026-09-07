// Pins every thread of the process to the fast CPU cores.
//
// Measured on the SM8650 reference device (runs/device_debug/automated/
// 20260904_172741/): the scheduler parks the inference worker on the little
// cluster under load, inflating parent inference by ~10%.  Pinning the whole
// process to the big/prime cores (cpu2-7) improved effective FPS from 7.05 to
// 7.66.  The fast cores are detected per device from cpufreq (>= 85% of the
// highest core frequency) instead of hardcoding a mask.  Any failure is
// non-fatal: the caller logs the returned core count and continues.

#include <jni.h>

#include <android/log.h>

#include <dirent.h>
#include <sched.h>

#include <cstdlib>
#include <string>
#include <vector>

#define LOG_TAG "TinyHeatmapNet"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace {

constexpr int kMaxCores = 32;
// A core counts as "fast" at >= 85% of the highest core frequency.  On the
// SM8650 reference device this selects cpu2-7 (2.96-3.30 GHz) and excludes
// the 2.27 GHz little pair, matching the measured experiment mask 0xFC.
constexpr long kFastCorePercent = 85;

long readMaxFreq(int cpu) {
    const std::string path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq";
    FILE* handle = fopen(path.c_str(), "r");
    if (handle == nullptr) return -1;
    long freq = -1;
    if (fscanf(handle, "%ld", &freq) != 1) freq = -1;
    fclose(handle);
    return freq;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_com_cupet_fallingball_vision_CpuAffinity_nativePinProcessToFastCores(JNIEnv* /*env*/,
                                                                          jobject /*thiz*/) {
    std::vector<int> freqs;
    long maxFreq = 0;
    for (int cpu = 0; cpu < kMaxCores; ++cpu) {
        const long freq = readMaxFreq(cpu);
        if (freq < 0) break;  // cores are contiguous; stop at the first missing one
        freqs.push_back(freq);
        if (freq > maxFreq) maxFreq = freq;
    }
    if (maxFreq <= 0) {
        LOGW("CpuAffinity: cpufreq unavailable; process stays on default affinity");
        return 0;
    }
    cpu_set_t fastSet;
    CPU_ZERO(&fastSet);
    int fastCount = 0;
    for (int cpu = 0; cpu < static_cast<int>(freqs.size()); ++cpu) {
        if (freqs[cpu] * 100 >= maxFreq * kFastCorePercent) {
            CPU_SET(cpu, &fastSet);
            ++fastCount;
        }
    }
    if (fastCount == 0) return 0;

    int pinnedThreads = 0;
    DIR* tasks = opendir("/proc/self/task");
    if (tasks != nullptr) {
        while (dirent* entry = readdir(tasks)) {
            if (entry->d_name[0] < '0' || entry->d_name[0] > '9') continue;
            const pid_t tid = static_cast<pid_t>(atoi(entry->d_name));
            if (sched_setaffinity(tid, sizeof(fastSet), &fastSet) == 0) ++pinnedThreads;
        }
        closedir(tasks);
    }
    LOGI("CpuAffinity: pinned %d threads to %d fast cores", pinnedThreads, fastCount);
    return fastCount;
}
