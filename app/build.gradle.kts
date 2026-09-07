import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

private fun readAliyunConfig(file: File): Map<String, String> =
    if (!file.isFile) emptyMap() else file.readLines()
        .mapNotNull { line ->
            val separator = line.indexOf(',')
            if (separator <= 0) null else {
                val key = line.substring(0, separator).trim().trim('"')
                val value = line.substring(separator + 1).trim().trim('"')
                key to value
            }
        }
        .toMap()

// Optional local AI credentials. Keep their contents outside version control.
val localConfig = Properties().apply {
    rootProject.file("local.properties").takeIf { it.isFile }?.inputStream()?.use { load(it) }
}
val aliyunConfigFile = rootProject.file(
    providers.gradleProperty("aliyunApiKeyCsv")
        .orElse(providers.environmentVariable("ALIYUN_API_KEY_CSV"))
        .orElse(localConfig.getProperty("aliyunApiKeyCsv") ?: ".secrets/aliyun-api-key.csv")
        .get()
)
val aliyunConfig = readAliyunConfig(aliyunConfigFile)

// QNN runtime objects are obtained from a local SDK and distributed only as part
// of the APK. Existing local copies remain usable when QNN_SDK_ROOT is unset.
val localNativeRuntimeDir = file("src/main/jniLibs/arm64-v8a")
val qnnRuntimeSdkPaths = listOf(
    "lib/aarch64-android/libQnnHtp.so",
    "lib/aarch64-android/libQnnHtpV75Stub.so",
    "lib/aarch64-android/libQnnHtpV75CalculatorStub.so",
    "lib/aarch64-android/libQnnSystem.so",
    "lib/hexagon-v75/unsigned/libQnnHtpV75.so",
    "lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so"
)
val qnnSdkRoot = providers.environmentVariable("QNN_SDK_ROOT")
    .map { it.trim() }
    .orElse("")
val nativeRuntimeFiles = qnnSdkRoot.map { sdkRoot ->
    val qnnFiles = qnnRuntimeSdkPaths.map { relativePath ->
        if (sdkRoot.isEmpty()) {
            localNativeRuntimeDir.resolve(relativePath.substringAfterLast('/'))
        } else {
            File(sdkRoot, relativePath)
        }
    }
    val missingQnnFiles = qnnFiles.filterNot { it.isFile }
    if (missingQnnFiles.isNotEmpty()) {
        throw GradleException(
            "QNN runtime libraries are missing. Set QNN_SDK_ROOT to your QAIRT/QNN " +
                "SDK 2.50 directory containing lib/aarch64-android and " +
                "lib/hexagon-v75/unsigned, or restore all six local QNN libraries in " +
                "app/src/main/jniLibs/arm64-v8a. See README.md. Missing files:\n" +
                missingQnnFiles.joinToString("\n") { it.absolutePath }
        )
    }
    val ncnnFile = localNativeRuntimeDir.resolve("libncnn.so")
    if (!ncnnFile.isFile) {
        throw GradleException(
            "NCNN runtime library is missing: ${ncnnFile.absolutePath}. " +
                "Restore app/src/main/jniLibs/arm64-v8a/libncnn.so from the repository."
        )
    }
    qnnFiles + ncnnFile
}
val localQnnSdkDir = file("src/main/cpp/third_party/qnn")
val qnnNoticeFiles = qnnSdkRoot.map { sdkRoot ->
    val noticeDir = if (sdkRoot.isEmpty()) localQnnSdkDir else File(sdkRoot)
    listOf("NOTICE.txt", "QNN_NOTICE.txt", "LICENSE.pdf")
        .map { noticeDir.resolve(it) }
        .filter { it.isFile }
}
val generatedNativeRuntimeDir = layout.buildDirectory.dir("generated/nativeRuntime")
val prepareNativeRuntime by tasks.registering(Sync::class) {
    group = "build"
    description = "Prepare local native libraries and third-party notices for APK packaging."
    from(nativeRuntimeFiles) {
        into("jniLibs/arm64-v8a")
    }
    from(qnnNoticeFiles) {
        into("assets/third_party/qnn")
    }
    from("src/main/cpp/third_party/ncnn") {
        include("LICENSE", "LICENSES/**")
        into("assets/third_party/ncnn")
    }
    into(generatedNativeRuntimeDir)
}

android {
    namespace = "com.cupet.fallingball"
    compileSdk = 35
    ndkVersion = "27.0.12077973"

    sourceSets.getByName("main").apply {
        jniLibs.setSrcDirs(listOf(prepareNativeRuntime.map { it.destinationDir.resolve("jniLibs") }))
        assets.srcDir(prepareNativeRuntime.map { it.destinationDir.resolve("assets") })
    }

    packaging {
        // The QNN HTP runtime (libQnnHtp.so) resolves its Hexagon stub libraries by
        // soname from the native library directory, which only exists when native
        // libs are extracted at install time.
        jniLibs {
            useLegacyPackaging = true
            // Hexagon DSP-side ELF objects are not ARM binaries; skip AGP stripping.
            doNotStrip += "**/libQnnHtpV75.so"
            doNotStrip += "**/libQnnHtpV75Skel.so"
        }
    }

    defaultConfig {
        applicationId = "com.cupet.fallingball"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        resValue("string", "aliyun_api_key", aliyunConfig["apiKey"].orEmpty())
        resValue(
            "string",
            "aliyun_openai_base_url",
            aliyunConfig["openAiCompatible"]
                ?: "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        resValue("string", "aliyun_workspace_id", aliyunConfig["workspaceId"].orEmpty())

        ndk {
            // The target Xiaomi 14 Ultra and the app's native pipeline are ARM64.
            abiFilters += "arm64-v8a"
        }

    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
    testOptions {
        unitTests.isIncludeAndroidResources = true
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

tasks.named("preBuild") {
    dependsOn(prepareNativeRuntime)
}
tasks.configureEach {
    // CMake imports NCNN from the generated directory, including during IDE sync.
    if (name.startsWith("configureCMake")) {
        dependsOn(prepareNativeRuntime)
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.recyclerview:recyclerview:1.4.0")
    implementation("androidx.work:work-runtime-ktx:2.10.0")
    implementation("androidx.dynamicanimation:dynamicanimation:1.0.0")

    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("androidx.concurrent:concurrent-futures:1.1.0")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("com.squareup.okhttp3:okhttp-jvm:5.3.0")

    // Keep the phone heatmap-C implementation on the same OpenCV release as the PC lab.
    implementation("org.opencv:opencv:4.12.0")

    // Standard Android ORT includes NNAPI; the QNN-only artifact does not.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")

    // JVM unit tests for the pure-algorithm layers (signal/, analysis/, geometry).
    // Those packages must stay free of Android imports so they run without a device.
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    testImplementation("com.squareup.okhttp3:mockwebserver:5.3.0")
    testImplementation("androidx.test:core:1.6.1")
    testImplementation("org.robolectric:robolectric:4.14.1")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
