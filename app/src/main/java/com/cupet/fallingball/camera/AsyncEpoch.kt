package com.cupet.fallingball.camera

import java.util.concurrent.atomic.AtomicLong

/** Invalidates asynchronous work without conflating it with the selected model profile. */
internal class AsyncEpoch {
    private val value = AtomicLong(0L)

    fun capture(): Long = value.get()

    fun invalidate(): Long = value.incrementAndGet()

    fun isCurrent(captured: Long): Boolean = value.get() == captured
}
