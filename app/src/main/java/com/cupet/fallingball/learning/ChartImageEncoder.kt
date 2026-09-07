package com.cupet.fallingball.learning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.util.Base64
import com.cupet.fallingball.analysis.MeasurementRecord
import com.cupet.fallingball.ui.DisplacementChartRenderer
import java.io.ByteArrayOutputStream

object ChartImageEncoder {
    const val WIDTH = 1_024
    const val HEIGHT = 640

    fun encodeRecord(context: Context, record: MeasurementRecord): String {
        val bitmap = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888)
        return try {
            DisplacementChartRenderer(context.resources).draw(
                Canvas(bitmap), WIDTH, HEIGHT, record
            )
            val bytes = ByteArrayOutputStream()
            check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, bytes)) {
                "Unable to encode chart"
            }
            Base64.encodeToString(bytes.toByteArray(), Base64.NO_WRAP)
        } finally {
            bitmap.recycle()
        }
    }
}
