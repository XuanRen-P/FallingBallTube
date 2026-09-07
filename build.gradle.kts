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
                "See the device data protection note in README.md."
        )
    }
}
