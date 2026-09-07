package com.cupet.fallingball.video

import java.util.concurrent.ExecutionException
import java.util.concurrent.Future

/** Keeps asynchronous notification/progress failures observable without blocking a video frame. */
internal class PendingWorkerUpdates {
    private val pending = ArrayDeque<Future<*>>()

    fun track(future: Future<*>) {
        // Retain every outstanding update: replacing only the latest Future can hide a failed
        // foreground-service request when a subsequent notification update succeeds.
        pending.addLast(future)
        checkCompleted()
    }

    fun checkCompleted() {
        val iterator = pending.iterator()
        while (iterator.hasNext()) {
            val future = iterator.next()
            if (future.isDone) {
                iterator.remove()
                await(future)
            }
        }
    }

    /** All WorkManager updates must settle before Worker returns a Result, including failures. */
    fun awaitAll() {
        var failure: Throwable? = null
        while (pending.isNotEmpty()) {
            val future = pending.removeFirst()
            try {
                await(future)
            } catch (error: Throwable) {
                if (failure == null) failure = error
                else if (failure !== error) failure.addSuppressed(error)
            }
        }
        failure?.let { throw it }
    }

    private fun await(future: Future<*>) {
        var interrupted = false
        try {
            while (true) {
                try {
                    future.get()
                    return
                } catch (_: InterruptedException) {
                    // Cancellation may interrupt Worker while notification I/O is pending.
                    // Preserve the interrupt, but settle this update before signalling finish.
                    interrupted = true
                } catch (error: ExecutionException) {
                    throw error.cause ?: error
                }
            }
        } finally {
            if (interrupted) Thread.currentThread().interrupt()
        }
    }
}
