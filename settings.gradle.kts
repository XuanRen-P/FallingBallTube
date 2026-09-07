pluginManagement {
    repositories {
        google()
        mavenCentral {
            // OpenCV 4.12's published Gradle module metadata contains stale AAR size/hash
            // values; its Maven POM and artifact checksums are correct.
            metadataSources {
                mavenPom()
                artifact()
            }
        }
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenCentral()
        google()
    }
}
rootProject.name = "FallingBallTube"
include(":app")
