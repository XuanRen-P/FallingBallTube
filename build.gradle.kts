plugins {
    id("com.android.application") version "8.7.2" apply false
    id("org.jetbrains.kotlin.android") version "2.2.21" apply false
}

gradle.taskGraph.whenReady {
    val forbiddenDeviceTest = allTasks.firstOrNull { task ->
        task.name.matches(Regex("^(connected|device).*AndroidTest$"))
    }
    if (forbiddenDeviceTest != null) {
        throw GradleException(
            "${forbiddenDeviceTest.path} is blocked because it may uninstall " +
                "com.cupet.fallingball and erase real-device data. " +
                "Use the manual instrumentation workflow in " +
                "docs/ANDROID_DEVICE_TESTING_INCIDENT_2026-08-14.md."
        )
    }
}
